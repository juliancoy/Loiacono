#include "loiacono_rolling.h"
#include "loiacono_gpu_compute.h"
#include "loiacono_parallel.h"
#include <algorithm>

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

    for (int i = 0; i < numBins; i++) {
        double fHz = std::exp(logMin + i * logStep);
        double fNorm = fHz / sampleRate;
        freqs_[i] = fNorm;

        int wlen = static_cast<int>(multiple / fNorm);
        wlen = std::min(wlen, RING_SIZE);
        windowLens_[i] = wlen;
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

        double angle = TWO_PI * f * static_cast<double>(sampleCount_);
        Tr_[fi] += sample * std::cos(angle) * norm;
        Ti_[fi] -= sample * std::sin(angle) * norm;

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
            if (gpuRollingCompute_ && gpuRollingCompute_->available()) {
                const auto& latest = pendingGpuChunks_.back();
                gpuRollingCompute_->processChunk(
                    latest.newSamples.data(),
                    count,
                    startSampleCount,
                    startRingHead);
            }
        } else if (mode == ComputeMode::SingleThread) {
            for (int i = 0; i < count; i++) {
                processSample(samples[i]);
            }
        } else {
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
                Ti_);
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
        if (gpuRollingCompute_ && gpuRollingCompute_->available() &&
            gpuRollingCompute_->spectrum(out)) {
            return;
        }
        if (gpuCompute_ && gpuCompute_->compute(ring_, static_cast<unsigned int>(ringHead_), out)) {
            return;
        }
    }
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
