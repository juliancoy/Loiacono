// Loiacono Rolling-Window Spectrogram
// 
// This implements a rolling-window variant of the Goertzel algorithm for
// efficient time-frequency analysis. Unlike the standard DFT/FFT which
// processes fixed blocks, this approach maintains running sums that are
// updated incrementally as new samples arrive.
//
// Key features:
// - Constant-Q frequency bins (logarithmically spaced, with window sizes
//   inversely proportional to frequency)
// - Goertzel-based complex coefficient computation with proper amplitude
//   scaling (1/sqrt(window_size) normalization)
// - Rolling window: subtracts samples as they leave the window, providing
//   sliding-window spectral estimates without block artifacts
// - Multiple compute modes: single-threaded, multi-threaded, and GPU-accelerated
//
// The algorithm is particularly efficient when many frequency bins are needed
// at fine time resolution, as each bin is computed independently with O(1)
// update per sample.

#include "loiacono_rolling.h"
#include "loiacono_gpu_compute.h"
#include "loiacono_parallel.h"
#include <algorithm>
#include <QDebug>

static constexpr double TWO_PI = 2.0 * M_PI;

LoiaconoRolling::~LoiaconoRolling() = default;

LoiaconoRolling::ComputeMode LoiaconoRolling::activeComputeMode() const
{
    if (computeMode_ == ComputeMode::GpuCompute && !gpuComputeAvailable()) {
        return ComputeMode::MultiThread;
    }
    return computeMode_;
}

const char* LoiaconoRolling::computeModeName(ComputeMode mode)
{
    switch (mode) {
    case ComputeMode::SingleThread:
        return "single-thread";
    case ComputeMode::MultiThread:
        return "multi-thread";
    case ComputeMode::GpuCompute:
        return "gpu-compute";
    }
    return "unknown";
}

bool LoiaconoRolling::gpuComputeAvailable() const
{
    return (gpuRollingCompute_ && gpuRollingCompute_->available()) ||
           (gpuCompute_ && gpuCompute_->available());
}

void LoiaconoRolling::configure(double sampleRate, double freqMin, double freqMax,
                                 int numBins, int multiple)
{
    std::lock_guard<std::mutex> lock(mutex_);

    sampleRate_ = sampleRate;
    multiple_ = multiple;
    numBins_ = numBins;

    freqs_.resize(numBins);
    windowLens_.resize(numBins);
    norms_.resize(numBins);

    double logMin = std::log(freqMin);
    double logMax = std::log(freqMax);
    double logStep = (logMax - logMin) / numBins;

    // Configure bins with Goertzel scaling:
    // - Window length scales inversely with frequency (constant Q)
    // - Normalization scales as 1/sqrt(window_length) for amplitude consistency
    for (int i = 0; i < numBins; i++) {
        double fHz = std::exp(logMin + i * logStep);
        double fNorm = fHz / sampleRate;
        freqs_[i] = fNorm;

        int wlen = static_cast<int>(multiple / fNorm);
        wlen = std::min(wlen, RING_SIZE);
        windowLens_[i] = wlen;
        // Goertzel amplitude scaling: 1/sqrt(N) to normalize for window size
        norms_[i] = 1.0 / std::sqrt(static_cast<double>(wlen));
    }

    Tr_.assign(numBins, 0.0);
    Ti_.assign(numBins, 0.0);
    ring_.assign(RING_SIZE, 0.0f);
    ringHead_ = 0;
    sampleCount_ = 0;
    pendingGpuChunks_.clear();
    pendingGpuChunksOverflowed_ = false;

    if (!gpuCompute_) {
        gpuCompute_ = std::make_unique<LoiaconoGpuCompute>();
    }
    if (!gpuRollingCompute_) {
        gpuRollingCompute_ = std::make_unique<LoiaconoGpuRollingCompute>();
    }
    gpuRollingCompute_->configure(RING_SIZE, 2048, numBins_, freqs_, norms_, windowLens_);
    gpuCompute_->configure(RING_SIZE, numBins_, multiple_, freqs_);
}

void LoiaconoRolling::processSample(float sample)
{
    ring_[ringHead_] = sample;

    for (int fi = 0; fi < numBins_; fi++) {
        double f = freqs_[fi];
        double norm = norms_[fi];
        int wlen = windowLens_[fi];

        // Apply leakiness factor per sample (correct for single-sample processing)
        Tr_[fi] *= leakiness_;
        Ti_[fi] *= leakiness_;

        // Rolling Goertzel recurrence:
        // The Goertzel algorithm computes DFT coefficients efficiently using a
        // second-order IIR filter. We use a complex-valued form that maintains
        // (Tr, Ti) as the accumulated complex Fourier coefficient.
        // 
        // For each sample x[n] at time n with normalized frequency f:
        //   s[n] = x[n] + 2*cos(2πf)*s[n-1] - s[n-2]  (standard recurrence)
        //   y[n] = s[n] - exp(-j*2πf)*s[n-1]          (output)
        //
        // Our implementation uses a direct complex accumulation:
        //   Tr += x[n] * cos(2πf*n) * norm  (real part with amplitude scaling)
        //   Ti -= x[n] * sin(2πf*n) * norm  (imaginary part with amplitude scaling)
        double angle = TWO_PI * f * static_cast<double>(sampleCount_);
        Tr_[fi] += sample * std::cos(angle) * norm;
        Ti_[fi] -= sample * std::sin(angle) * norm;

        // Rolling window: subtract the sample that just left the window
        if (sampleCount_ >= static_cast<uint64_t>(wlen)) {
            int oldIdx = (ringHead_ - wlen + RING_SIZE) % RING_SIZE;
            float oldSample = ring_[oldIdx];
            double oldAngle = TWO_PI * f * static_cast<double>(sampleCount_ - wlen);
            Tr_[fi] -= oldSample * std::cos(oldAngle) * norm;
            Ti_[fi] += oldSample * std::sin(oldAngle) * norm;
        }
    }

    ringHead_ = (ringHead_ + 1) % RING_SIZE;
    sampleCount_++;
}

void LoiaconoRolling::processChunk(const float* samples, int count)
{
    auto t0 = Clock::now();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        ComputeMode mode = activeComputeMode();
        if (mode == ComputeMode::GpuCompute) {
            const int startRingHead = ringHead_;
            const uint64_t startSampleCount = sampleCount_;
            for (int i = 0; i < count; i++) {
                ring_[(startRingHead + i) % RING_SIZE] = samples[i];
            }
            ringHead_ = (startRingHead + count) % RING_SIZE;
            sampleCount_ += count;
            GpuChunkDelta delta;
            delta.newSamples.assign(samples, samples + count);
            delta.ringHeadStart = startRingHead;
            delta.startSampleCount = startSampleCount;
            pendingGpuChunks_.push_back(std::move(delta));
            while (pendingGpuChunks_.size() > MAX_PENDING_GPU_CHUNKS) {
                pendingGpuChunks_.pop_front();
                pendingGpuChunksOverflowed_ = true;
            }
            // GPU compute will be processed in the GUI thread via takePendingGpuChunks()
            // Don't call gpuRollingCompute_->processChunk() here as it uses OpenGL
            // from the audio thread which causes threading issues
        } else if (mode == ComputeMode::SingleThread) {
            for (int i = 0; i < count; i++) {
                processSample(samples[i]);
            }
        } else {
            // Multi-thread mode
            const int startRingHead = ringHead_;
            const uint64_t startSampleCount = sampleCount_;

            for (int i = 0; i < count; i++) {
                ring_[(startRingHead + i) % RING_SIZE] = samples[i];
            }
            ringHead_ = (startRingHead + count) % RING_SIZE;
            sampleCount_ += count;

            loiacono::processBinsParallel(
                workerCount_,
                numBins_,
                count,
                startRingHead,
                startSampleCount,
                samples,
                ring_,
                freqs_,
                norms_,
                windowLens_,
                Tr_,
                Ti_,
                leakiness_);
        }
    }

    auto elapsed = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    chunkCount_++;
    totalChunkMicros_ += elapsed;
    lastChunkSamples_ = count;
    if (elapsed > peakChunkMicros_) peakChunkMicros_ = elapsed;
}

void LoiaconoRolling::getSpectrum(std::vector<float>& out) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (activeComputeMode() == ComputeMode::GpuCompute) {
        // In CPU display mode, we need to process pending chunks before reading spectrum
        // because the GPU paint path won't be called
        if (!pendingGpuChunks_.empty() && gpuRollingCompute_ && gpuRollingCompute_->available()) {
            // Take chunks and process them (need to cast away const for modification)
            auto& chunks = const_cast<std::deque<GpuChunkDelta>&>(pendingGpuChunks_);
            auto& overflowed = const_cast<bool&>(pendingGpuChunksOverflowed_);
            
            while (!chunks.empty()) {
                const auto& delta = chunks.front();
                gpuRollingCompute_->processChunk(delta.newSamples.data(), 
                                                 static_cast<int>(delta.newSamples.size()),
                                                 delta.startSampleCount,
                                                 delta.ringHeadStart,
                                                 leakiness_);
                chunks.pop_front();
            }
            overflowed = false;
        }
        
        if (gpuRollingCompute_ && gpuRollingCompute_->available()) {
            if (gpuRollingCompute_->spectrum(out)) {
                return;
            }
        }
        
        if (gpuCompute_ && gpuCompute_->compute(ring_, static_cast<unsigned int>(ringHead_), out)) {
            return;
        }
    }
    
    // CPU fallback (used by SingleThread and MultiThread modes)
    out.resize(numBins_);
    for (int i = 0; i < numBins_; i++) {
        out[i] = static_cast<float>(std::sqrt(Tr_[i] * Tr_[i] + Ti_[i] * Ti_[i]));
    }
}

LoiaconoRolling::Stats LoiaconoRolling::getStats() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    Stats s;
    s.totalSamples = sampleCount_;
    s.totalChunks = chunkCount_;
    auto now = Clock::now();
    s.uptimeSeconds = std::chrono::duration<double>(now - startTime_).count();
    s.samplesPerSecond = s.uptimeSeconds > 0 ? sampleCount_ / s.uptimeSeconds : 0;
    s.avgChunkMicros = chunkCount_ > 0 ? totalChunkMicros_ / chunkCount_ : 0;
    s.peakChunkMicros = peakChunkMicros_;
    // CPU load: how much of the audio deadline does processChunk use?
    // deadline = chunkSamples / sampleRate (in micros)
    double deadlineMicros = (lastChunkSamples_ / sampleRate_) * 1e6;
    s.cpuLoadPercent = deadlineMicros > 0 ? (s.avgChunkMicros / deadlineMicros) * 100.0 : 0;
    s.currentBins = numBins_;
    s.currentMultiple = multiple_;
    s.freqMin = numBins_ > 0 ? freqs_[0] * sampleRate_ : 0;
    s.freqMax = numBins_ > 0 ? freqs_[numBins_ - 1] * sampleRate_ : 0;
    return s;
}

LoiaconoRolling::GpuInputSnapshot LoiaconoRolling::gpuInputSnapshot() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    GpuInputSnapshot snapshot;
    snapshot.ring = ring_;
    snapshot.freqs = freqs_;
    snapshot.norms = norms_;
    snapshot.windowLens = windowLens_;
    snapshot.offset = static_cast<unsigned int>(ringHead_);
    snapshot.sampleCount = sampleCount_;
    snapshot.numBins = numBins_;
    snapshot.multiple = multiple_;
    return snapshot;
}

LoiaconoRolling::GpuChunkBatch LoiaconoRolling::takePendingGpuChunks()
{
    std::lock_guard<std::mutex> lock(mutex_);
    GpuChunkBatch batch;
    batch.overflowed = pendingGpuChunksOverflowed_;
    batch.chunks.reserve(pendingGpuChunks_.size());
    while (!pendingGpuChunks_.empty()) {
        batch.chunks.push_back(std::move(pendingGpuChunks_.front()));
        pendingGpuChunks_.pop_front();
    }
    pendingGpuChunksOverflowed_ = false;
    return batch;
}
