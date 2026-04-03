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
#include <array>
#include <future>
#include <QDebug>

static constexpr double TWO_PI = 2.0 * M_PI;

LoiaconoRolling::~LoiaconoRolling() = default;

void LoiaconoRolling::setComputeMode(ComputeMode mode)
{
    std::lock_guard<std::mutex> lock(mutex_);
    computeMode_ = mode;
    if ((computeMode_ == ComputeMode::GpuCompute || computeMode_ == ComputeMode::VulkanCompute) && numBins_ > 0) {
        ensureGpuBackendsConfiguredLocked();
    }
}

LoiaconoRolling::ComputeMode LoiaconoRolling::activeComputeMode() const
{
    if (computeMode_ == ComputeMode::GpuCompute && !gpuComputeAvailable()) {
        return ComputeMode::MultiThread;
    }
    if (computeMode_ == ComputeMode::VulkanCompute && !vulkanComputeAvailable()) {
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
    case ComputeMode::VulkanCompute:
        return "vulkan-compute";
    }
    return "unknown";
}

const char* LoiaconoRolling::windowModeName(WindowMode mode)
{
    switch (mode) {
    case WindowMode::RectangularWindow:
        return "rectangular";
    case WindowMode::HannWindow:
        return "hann";
    case WindowMode::HammingWindow:
        return "hamming";
    case WindowMode::BlackmanWindow:
        return "blackman";
    case WindowMode::BlackmanHarrisWindow:
        return "blackman-harris";
    case WindowMode::LeakyWindow:
        return "leaky";
    }
    return "unknown";
}

const char* LoiaconoRolling::windowLengthModeName(WindowLengthMode mode)
{
    switch (mode) {
    case WindowLengthMode::ConstantSamples:
        return "constant";
    case WindowLengthMode::SqrtPeriod:
        return "sqrt";
    case WindowLengthMode::PeriodMultiple:
        return "periods";
    }
    return "unknown";
}

const char* LoiaconoRolling::algorithmModeName(AlgorithmMode mode)
{
    switch (mode) {
    case AlgorithmMode::Loiacono:
        return "loiacono";
    case AlgorithmMode::FFT:
        return "fft";
    case AlgorithmMode::Goertzel:
        return "goertzel";
    }
    return "unknown";
}

const char* LoiaconoRolling::normalizationModeName(NormalizationMode mode)
{
    switch (mode) {
    case NormalizationMode::RawSum:
        return "raw";
    case NormalizationMode::CoherentAmplitude:
        return "coherent";
    case NormalizationMode::Energy:
        return "energy";
    }
    return "unknown";
}

bool LoiaconoRolling::gpuComputeAvailable() const
{
    return gpuCompute_ && gpuCompute_->available();
}

bool LoiaconoRolling::vulkanComputeAvailable() const
{
    return vulkanCompute_ && vulkanCompute_->available();
}

double LoiaconoRolling::normalizationScaleForWindow(int wlen) const
{
    const int safeLen = std::max(1, wlen);
    double weightSum = 0.0;
    double weightEnergy = 0.0;
    for (int i = 0; i < safeLen; ++i) {
        double w = windowWeightAt(i, safeLen);
        weightSum += w;
        weightEnergy += w * w;
    }
    switch (normalizationMode_) {
    case NormalizationMode::RawSum:
        return 1.0;
    case NormalizationMode::CoherentAmplitude:
        return weightSum > 0.0 ? 1.0 / weightSum : 1.0;
    case NormalizationMode::Energy:
        return weightEnergy > 0.0 ? 1.0 / std::sqrt(weightEnergy) : 1.0;
    }
    return 1.0;
}

double LoiaconoRolling::effectiveLeakiness() const
{
    return windowMode_ == WindowMode::LeakyWindow ? leakiness_ : 1.0;
}

bool LoiaconoRolling::usesRollingState() const
{
    if (algorithmMode_ != AlgorithmMode::Loiacono) return false;
    return windowMode_ == WindowMode::RectangularWindow || windowMode_ == WindowMode::LeakyWindow;
}

double LoiaconoRolling::windowWeightAt(int index, int wlen) const
{
    if (wlen <= 1) return 1.0;
    const double phase = TWO_PI * static_cast<double>(index) / static_cast<double>(wlen - 1);
    switch (windowMode_) {
    case WindowMode::RectangularWindow:
    case WindowMode::LeakyWindow:
        return 1.0;
    case WindowMode::HannWindow:
        return 0.5 - 0.5 * std::cos(phase);
    case WindowMode::HammingWindow:
        return 0.54 - 0.46 * std::cos(phase);
    case WindowMode::BlackmanWindow:
        return 0.42 - 0.5 * std::cos(phase) + 0.08 * std::cos(2.0 * phase);
    case WindowMode::BlackmanHarrisWindow:
        return 0.35875 - 0.48829 * std::cos(phase) + 0.14128 * std::cos(2.0 * phase)
            - 0.01168 * std::cos(3.0 * phase);
    }
    return 1.0;
}

const std::vector<float>& LoiaconoRolling::cachedWindowWeights(int wlen) const
{
    auto it = cachedWindowWeights_.find(wlen);
    if (it != cachedWindowWeights_.end()) {
        return it->second;
    }

    std::vector<float> weights(static_cast<size_t>(std::max(1, wlen)), 1.0f);
    if (wlen > 1) {
        for (int i = 0; i < wlen; ++i) {
            weights[static_cast<size_t>(i)] = static_cast<float>(windowWeightAt(i, wlen));
        }
    }
    auto [inserted, ok] = cachedWindowWeights_.emplace(wlen, std::move(weights));
    return inserted->second;
}

bool LoiaconoRolling::ensureGpuBackendsConfiguredLocked()
{
    if (!gpuCompute_) {
        gpuCompute_ = std::make_unique<LoiaconoGpuCompute>();
    }
    if (!gpuRollingCompute_) {
        gpuRollingCompute_ = std::make_unique<LoiaconoGpuRollingCompute>();
    }
    if (!vulkanCompute_) {
        vulkanCompute_ = std::make_unique<LoiaconoVulkanCompute>();
    }

    bool rollingOk = gpuRollingCompute_->configure(RING_SIZE, 2048, numBins_, freqs_, norms_, windowLens_);
    const int fftLength = fftWindowLengthForCurrentConfig();
    bool computeOk = gpuCompute_->configure(RING_SIZE,
                                            numBins_,
                                            freqs_,
                                            norms_,
                                            windowLens_,
                                            static_cast<int>(algorithmMode_),
                                            static_cast<int>(windowMode_),
                                            static_cast<int>(normalizationMode_),
                                            fftLength);
    bool vulkanOk = vulkanCompute_->configure(RING_SIZE,
                                              numBins_,
                                              freqs_,
                                              norms_,
                                              windowLens_,
                                              static_cast<int>(algorithmMode_),
                                              static_cast<int>(windowMode_),
                                              static_cast<int>(normalizationMode_),
                                              fftLength);
    return rollingOk || computeOk || vulkanOk;
}

void LoiaconoRolling::computeSpectrumFromRingLocked(std::vector<float>& out) const
{
    SpectrumSnapshot snapshot;
    snapshot.ring = ring_;
    snapshot.freqs = freqs_;
    snapshot.norms = norms_;
    snapshot.windowLens = windowLens_;
    snapshot.ringHead = ringHead_;
    snapshot.sampleCount = sampleCount_;
    snapshot.windowMode = windowMode_;
    snapshot.algorithmMode = algorithmMode_;
    snapshot.leakiness = effectiveLeakiness();
    computeSpectrumFromSnapshot(snapshot, out);
}

void LoiaconoRolling::computeSpectrumFromSnapshot(const SpectrumSnapshot& snapshot, std::vector<float>& out) const
{
    switch (snapshot.algorithmMode) {
    case AlgorithmMode::Loiacono:
        computeSpectrumLoiaconoFromSnapshot(snapshot, out);
        return;
    case AlgorithmMode::FFT:
        computeSpectrumFftFromSnapshot(snapshot, out);
        return;
    case AlgorithmMode::Goertzel:
        computeSpectrumGoertzelFromSnapshot(snapshot, out);
        return;
    }
    out.assign(snapshot.freqs.size(), 0.0f);
}

void LoiaconoRolling::computeSpectrumLoiaconoFromSnapshot(const SpectrumSnapshot& snapshot, std::vector<float>& out) const
{
    const int numBins = static_cast<int>(snapshot.freqs.size());
    out.resize(numBins);
    std::unordered_map<int, const std::vector<float>*> windowCache;
    if (snapshot.windowMode != WindowMode::RectangularWindow && snapshot.windowMode != WindowMode::LeakyWindow) {
        for (int fi = 0; fi < numBins; ++fi) {
            const int wlen = snapshot.windowLens[fi];
            const uint64_t startSample = snapshot.sampleCount > static_cast<uint64_t>(wlen)
                ? (snapshot.sampleCount - static_cast<uint64_t>(wlen))
                : 0;
            const int validSamples = static_cast<int>(snapshot.sampleCount - startSample);
            if (windowCache.find(validSamples) == windowCache.end()) {
                windowCache.emplace(validSamples, &cachedWindowWeights(validSamples));
            }
        }
    }

    auto processRange = [&](int begin, int end) {
        for (int fi = begin; fi < end; ++fi) {
            double tr = 0.0;
            double ti = 0.0;
            const double f = snapshot.freqs[fi];
            const double norm = snapshot.norms[fi];
            const int wlen = snapshot.windowLens[fi];
            const uint64_t startSample = snapshot.sampleCount > static_cast<uint64_t>(wlen)
                ? (snapshot.sampleCount - static_cast<uint64_t>(wlen))
                : 0;
            const int validSamples = static_cast<int>(snapshot.sampleCount - startSample);
            const int readOffset = (snapshot.ringHead - validSamples + RING_SIZE) % RING_SIZE;
            const std::vector<float>* weights = nullptr;
            if (snapshot.windowMode != WindowMode::RectangularWindow && snapshot.windowMode != WindowMode::LeakyWindow) {
                weights = windowCache.at(validSamples);
            }

            const double delta = TWO_PI * f;
            const double cosDelta = std::cos(delta);
            const double sinDelta = std::sin(delta);
            double cosAngle = std::cos(delta * static_cast<double>(startSample));
            double sinAngle = std::sin(delta * static_cast<double>(startSample));

            for (int k = 0; k < validSamples; ++k) {
                const int ringIdx = (readOffset + k) % RING_SIZE;
                const float sample = snapshot.ring[ringIdx];
                const double weight = weights ? (*weights)[static_cast<size_t>(k)] : 1.0;
                tr *= snapshot.leakiness;
                ti *= snapshot.leakiness;
                tr += sample * cosAngle * norm * weight;
                ti -= sample * sinAngle * norm * weight;

                const double nextCos = cosAngle * cosDelta - sinAngle * sinDelta;
                sinAngle = sinAngle * cosDelta + cosAngle * sinDelta;
                cosAngle = nextCos;
            }
            out[fi] = static_cast<float>(std::sqrt(tr * tr + ti * ti));
        }
    };

    unsigned int threads = std::min<unsigned int>(workerCount_, std::max(1, numBins));
    if (threads <= 1 || numBins < 64) {
        processRange(0, numBins);
        return;
    }

    std::vector<std::future<void>> jobs;
    jobs.reserve(threads);
    int binsPerThread = (numBins + static_cast<int>(threads) - 1) / static_cast<int>(threads);
    for (unsigned int t = 0; t < threads; ++t) {
        int begin = static_cast<int>(t) * binsPerThread;
        int end = std::min(numBins, begin + binsPerThread);
        if (begin >= end) break;
        jobs.push_back(std::async(std::launch::async, processRange, begin, end));
    }
    for (auto& job : jobs) {
        job.get();
    }
}

void LoiaconoRolling::computeSpectrumGoertzelFromSnapshot(const SpectrumSnapshot& snapshot, std::vector<float>& out) const
{
    const int numBins = static_cast<int>(snapshot.freqs.size());
    out.resize(numBins);
    std::unordered_map<int, const std::vector<float>*> windowCache;
    if (snapshot.windowMode != WindowMode::RectangularWindow && snapshot.windowMode != WindowMode::LeakyWindow) {
        for (int fi = 0; fi < numBins; ++fi) {
            const int wlen = snapshot.windowLens[fi];
            const uint64_t startSample = snapshot.sampleCount > static_cast<uint64_t>(wlen)
                ? (snapshot.sampleCount - static_cast<uint64_t>(wlen))
                : 0;
            const int validSamples = static_cast<int>(snapshot.sampleCount - startSample);
            if (windowCache.find(validSamples) == windowCache.end()) {
                windowCache.emplace(validSamples, &cachedWindowWeights(validSamples));
            }
        }
    }

    auto processRange = [&](int begin, int end) {
        for (int fi = begin; fi < end; ++fi) {
            const double f = snapshot.freqs[fi];
            const double norm = snapshot.norms[fi];
            const int wlen = snapshot.windowLens[fi];
            const uint64_t startSample = snapshot.sampleCount > static_cast<uint64_t>(wlen)
                ? (snapshot.sampleCount - static_cast<uint64_t>(wlen))
                : 0;
            const int validSamples = static_cast<int>(snapshot.sampleCount - startSample);
            const int readOffset = (snapshot.ringHead - validSamples + RING_SIZE) % RING_SIZE;
            const std::vector<float>* weights = nullptr;
            if (snapshot.windowMode != WindowMode::RectangularWindow && snapshot.windowMode != WindowMode::LeakyWindow) {
                weights = windowCache.at(validSamples);
            }

            const double omega = TWO_PI * f;
            const double coeff = 2.0 * std::cos(omega);
            double q0 = 0.0;
            double q1 = 0.0;
            double q2 = 0.0;
            for (int k = 0; k < validSamples; ++k) {
                const int ringIdx = (readOffset + k) % RING_SIZE;
                const float sample = snapshot.ring[ringIdx];
                const double weight = weights ? (*weights)[static_cast<size_t>(k)] : 1.0;
                q0 = sample * weight + coeff * q1 - q2;
                q2 = q1;
                q1 = q0;
            }
            const double real = q1 - q2 * std::cos(omega);
            const double imag = q2 * std::sin(omega);
            out[fi] = static_cast<float>(std::sqrt(real * real + imag * imag) * norm);
        }
    };

    unsigned int threads = std::min<unsigned int>(workerCount_, std::max(1, numBins));
    if (threads <= 1 || numBins < 64) {
        processRange(0, numBins);
        return;
    }
    std::vector<std::future<void>> jobs;
    jobs.reserve(threads);
    int binsPerThread = (numBins + static_cast<int>(threads) - 1) / static_cast<int>(threads);
    for (unsigned int t = 0; t < threads; ++t) {
        int begin = static_cast<int>(t) * binsPerThread;
        int end = std::min(numBins, begin + binsPerThread);
        if (begin >= end) break;
        jobs.push_back(std::async(std::launch::async, processRange, begin, end));
    }
    for (auto& job : jobs) {
        job.get();
    }
}

int LoiaconoRolling::fftWindowLengthForCurrentConfig() const
{
    int maxWindow = 2;
    for (int wlen : windowLens_) {
        maxWindow = std::max(maxWindow, wlen);
    }
    int fftLen = 1;
    while (fftLen * 2 <= maxWindow && fftLen * 2 <= RING_SIZE) {
        fftLen *= 2;
    }
    return std::max(2, fftLen);
}

std::vector<double> LoiaconoRolling::fftWindowWeights(int wlen) const
{
    std::vector<double> weights(static_cast<size_t>(std::max(1, wlen)), 1.0);
    for (int i = 0; i < wlen; ++i) {
        weights[static_cast<size_t>(i)] = windowWeightAt(i, wlen);
    }
    return weights;
}

void LoiaconoRolling::fftInPlace(std::vector<std::complex<double>>& data) const
{
    const int n = static_cast<int>(data.size());
    int j = 0;
    for (int i = 1; i < n; ++i) {
        int bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) std::swap(data[static_cast<size_t>(i)], data[static_cast<size_t>(j)]);
    }
    for (int len = 2; len <= n; len <<= 1) {
        const double angle = -TWO_PI / static_cast<double>(len);
        const std::complex<double> wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (int j2 = 0; j2 < len / 2; ++j2) {
                const auto u = data[static_cast<size_t>(i + j2)];
                const auto v = data[static_cast<size_t>(i + j2 + len / 2)] * w;
                data[static_cast<size_t>(i + j2)] = u + v;
                data[static_cast<size_t>(i + j2 + len / 2)] = u - v;
                w *= wlen;
            }
        }
    }
}

void LoiaconoRolling::computeSpectrumFftFromSnapshot(const SpectrumSnapshot& snapshot, std::vector<float>& out) const
{
    const int numBins = static_cast<int>(snapshot.freqs.size());
    out.assign(numBins, 0.0f);
    const int fftLen = fftWindowLengthForCurrentConfig();
    const int validSamples = std::min<int>(fftLen, static_cast<int>(std::min<uint64_t>(snapshot.sampleCount, static_cast<uint64_t>(fftLen))));
    if (validSamples <= 0) return;

    const auto weights = fftWindowWeights(validSamples);
    const double fftNorm = normalizationScaleForWindow(validSamples);
    std::vector<std::complex<double>> data(static_cast<size_t>(fftLen), std::complex<double>(0.0, 0.0));
    const int readOffset = (snapshot.ringHead - validSamples + RING_SIZE) % RING_SIZE;
    for (int k = 0; k < validSamples; ++k) {
        const int ringIdx = (readOffset + k) % RING_SIZE;
        data[static_cast<size_t>(k)] = std::complex<double>(snapshot.ring[ringIdx] * weights[static_cast<size_t>(k)], 0.0);
    }

    fftInPlace(data);

    const int nyquist = fftLen / 2;
    std::vector<double> magnitudes(static_cast<size_t>(nyquist + 1), 0.0);
    for (int i = 0; i <= nyquist; ++i) {
        magnitudes[static_cast<size_t>(i)] = std::abs(data[static_cast<size_t>(i)]) * fftNorm;
    }

    for (int fi = 0; fi < numBins; ++fi) {
        const double targetIndex = snapshot.freqs[fi] * fftLen;
        if (targetIndex <= 0.0) {
            out[fi] = static_cast<float>(magnitudes.front());
        } else if (targetIndex >= nyquist) {
            out[fi] = static_cast<float>(magnitudes.back());
        } else {
            const int i0 = static_cast<int>(std::floor(targetIndex));
            const int i1 = std::min(nyquist, i0 + 1);
            const double frac = targetIndex - static_cast<double>(i0);
            out[fi] = static_cast<float>(magnitudes[static_cast<size_t>(i0)] * (1.0 - frac)
                + magnitudes[static_cast<size_t>(i1)] * frac);
        }
    }
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
    double logStep = numBins > 1 ? (logMax - logMin) / (numBins - 1) : 0.0;

    // Configure bins with Goertzel scaling:
    // - Window length follows the selected frequency-to-window mapping
    // - Normalization scales as 1/sqrt(window_length) for amplitude consistency
    // - Window length is limited to RING_SIZE to prevent ring buffer overflow
    const double maxFreqNorm = std::exp(logMax) / sampleRate;
    const double safeMaxFreqNorm = std::max(maxFreqNorm, 1.0 / sampleRate);
    const double baseWindow = static_cast<double>(multiple) / safeMaxFreqNorm;
    double windowLengthExponent = 1.0;
    switch (windowLengthMode_) {
    case WindowLengthMode::ConstantSamples:
        windowLengthExponent = 0.0;
        break;
    case WindowLengthMode::SqrtPeriod:
        windowLengthExponent = 0.5;
        break;
    case WindowLengthMode::PeriodMultiple:
        windowLengthExponent = 1.0;
        break;
    }
    for (int i = 0; i < numBins; i++) {
        double fHz = std::exp(logMin + i * logStep);
        double fNorm = fHz / sampleRate;
        freqs_[i] = fNorm;

        const double freqRatio = std::clamp(safeMaxFreqNorm / std::max(fNorm, 1.0 / sampleRate), 1.0, static_cast<double>(RING_SIZE));
        int wlen = static_cast<int>(std::lround(baseWindow * std::pow(freqRatio, windowLengthExponent)));
        wlen = std::max(wlen, 2);
        wlen = std::min(wlen, RING_SIZE);
        windowLens_[i] = wlen;
        // Goertzel amplitude scaling: 1/sqrt(N) to normalize for window size
        norms_[i] = normalizationScaleForWindow(wlen);
    }

    Tr_.assign(numBins, 0.0);
    Ti_.assign(numBins, 0.0);
    ring_.assign(RING_SIZE, 0.0f);
    ringHead_ = 0;
    sampleCount_ = 0;
    pendingGpuChunks_.clear();
    pendingGpuChunksOverflowed_ = false;
    cachedWindowWeights_.clear();

    if (computeMode_ == ComputeMode::GpuCompute || computeMode_ == ComputeMode::VulkanCompute) {
        ensureGpuBackendsConfiguredLocked();
    }
}

void LoiaconoRolling::processSample(float sample)
{
    const double leak = effectiveLeakiness();
    float overwrittenSample = ring_[ringHead_];
    ring_[ringHead_] = sample;

    for (int fi = 0; fi < numBins_; fi++) {
        double f = freqs_[fi];
        double norm = norms_[fi];
        int wlen = windowLens_[fi];

        // Apply leakiness factor per sample (correct for single-sample processing)
        Tr_[fi] *= leak;
        Ti_[fi] *= leak;

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
            float oldSample = (oldIdx == ringHead_) ? overwrittenSample : ring_[oldIdx];
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
        const double leak = effectiveLeakiness();
        ComputeMode mode = activeComputeMode();
        if (mode == ComputeMode::GpuCompute || mode == ComputeMode::VulkanCompute || !usesRollingState()) {
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
            if (mode == ComputeMode::GpuCompute || mode == ComputeMode::VulkanCompute) {
                pendingGpuChunks_.push_back(std::move(delta));
                while (pendingGpuChunks_.size() > MAX_PENDING_GPU_CHUNKS) {
                    pendingGpuChunks_.pop_front();
                    pendingGpuChunksOverflowed_ = true;
                }
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
            std::vector<float> overwrittenSamples(count);

            for (int i = 0; i < count; i++) {
                overwrittenSamples[i] = ring_[(startRingHead + i) % RING_SIZE];
            }

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
                overwrittenSamples,
                leak);
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
    SpectrumSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const float leak = static_cast<float>(effectiveLeakiness());
        
        const auto activeMode = activeComputeMode();
        if (activeMode == ComputeMode::GpuCompute || activeMode == ComputeMode::VulkanCompute) {
            const unsigned int availableSamples = static_cast<unsigned int>(
                std::min<uint64_t>(sampleCount_, static_cast<uint64_t>(RING_SIZE)));
            if (activeMode == ComputeMode::VulkanCompute) {
                if (vulkanCompute_ && vulkanCompute_->compute(ring_,
                                                              static_cast<unsigned int>(ringHead_),
                                                              availableSamples,
                                                              leak,
                                                              out)) {
                    return;
                }
            } else {
                if (gpuCompute_ && gpuCompute_->compute(ring_,
                                                        static_cast<unsigned int>(ringHead_),
                                                        availableSamples,
                                                        leak,
                                                        out)) {
                    return;
                }
            }

            if (usesRollingState() && !pendingGpuChunks_.empty() && gpuRollingCompute_ && gpuRollingCompute_->available()) {
                auto& chunks = const_cast<std::deque<GpuChunkDelta>&>(pendingGpuChunks_);
                auto& overflowed = const_cast<bool&>(pendingGpuChunksOverflowed_);

                bool rollingOk = true;
                while (!chunks.empty()) {
                    const auto& delta = chunks.front();
                    rollingOk = gpuRollingCompute_->processChunk(delta.newSamples.data(),
                                                                 static_cast<int>(delta.newSamples.size()),
                                                                 delta.startSampleCount,
                                                                 delta.ringHeadStart,
                                                                 leak);
                    chunks.pop_front();
                    if (!rollingOk) break;
                }
                overflowed = false;

                if (rollingOk && gpuRollingCompute_->spectrum(out)) {
                    return;
                }
            }
        }
        
        if (computeMode_ == ComputeMode::GpuCompute || computeMode_ == ComputeMode::VulkanCompute || !usesRollingState()) {
            snapshot.ring = ring_;
            snapshot.freqs = freqs_;
            snapshot.norms = norms_;
            snapshot.windowLens = windowLens_;
            snapshot.ringHead = ringHead_;
            snapshot.sampleCount = sampleCount_;
            snapshot.computeMode = computeMode_;
            snapshot.windowMode = windowMode_;
            snapshot.algorithmMode = algorithmMode_;
            snapshot.leakiness = effectiveLeakiness();
            if (usesRollingState()) {
                snapshot.tr = Tr_;
                snapshot.ti = Ti_;
            }
        } else {
            out.resize(numBins_);
            for (int i = 0; i < numBins_; i++) {
                out[i] = static_cast<float>(std::sqrt(Tr_[i] * Tr_[i] + Ti_[i] * Ti_[i]));
            }
            return;
        }
    }

    computeSpectrumFromSnapshot(snapshot, out);
}

void LoiaconoRolling::getSpectraAtSampleCounts(const std::vector<uint64_t>& sampleCounts,
                                               std::vector<std::vector<float>>& out) const
{
    out.clear();
    if (sampleCounts.empty()) return;

    SpectrumSnapshot snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot.ring = ring_;
        snapshot.freqs = freqs_;
        snapshot.norms = norms_;
        snapshot.windowLens = windowLens_;
        snapshot.ringHead = ringHead_;
        snapshot.sampleCount = sampleCount_;
        snapshot.computeMode = computeMode_;
        snapshot.windowMode = windowMode_;
        snapshot.algorithmMode = algorithmMode_;
        snapshot.leakiness = effectiveLeakiness();
    }

    out.resize(sampleCounts.size());
    for (size_t i = 0; i < sampleCounts.size(); ++i) {
        SpectrumSnapshot historical = snapshot;
        const uint64_t clampedSampleCount = std::min(sampleCounts[i], snapshot.sampleCount);
        const uint64_t delta = snapshot.sampleCount - clampedSampleCount;
        if (delta >= static_cast<uint64_t>(RING_SIZE)) {
            historical.sampleCount = snapshot.sampleCount > static_cast<uint64_t>(RING_SIZE)
                ? (snapshot.sampleCount - static_cast<uint64_t>(RING_SIZE - 1))
                : 0;
            historical.ringHead = (snapshot.ringHead - (RING_SIZE - 1) + RING_SIZE) % RING_SIZE;
        } else {
            historical.sampleCount = clampedSampleCount;
            historical.ringHead = (snapshot.ringHead - static_cast<int>(delta) + RING_SIZE) % RING_SIZE;
        }
        computeSpectrumFromSnapshot(historical, out[i]);
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

// Convert bin index to frequency (Hz)
double LoiaconoRolling::binToFreqHz(double binIndex) const
{
    if (numBins_ <= 1) return binFreqHz(0);
    double t = std::clamp(binIndex / (numBins_ - 1.0), 0.0, 1.0);
    double logMin = std::log(freqs_[0] * sampleRate_);
    double logMax = std::log(freqs_[numBins_ - 1] * sampleRate_);
    return std::exp(logMin + t * (logMax - logMin));
}

// Convert frequency (Hz) to bin index (can be fractional)
double LoiaconoRolling::freqToBin(double freqHz) const
{
    if (numBins_ <= 1) return 0;
    double fMin = freqs_[0] * sampleRate_;
    double fMax = freqs_[numBins_ - 1] * sampleRate_;
    if (freqHz <= fMin) return 0;
    if (freqHz >= fMax) return numBins_ - 1;
    double logMin = std::log(fMin);
    double logMax = std::log(fMax);
    double logFreq = std::log(freqHz);
    double t = (logFreq - logMin) / (logMax - logMin);
    return t * (numBins_ - 1.0);
}

// Get interpolated spectrum value at fractional bin index
float LoiaconoRolling::interpolatedSpectrum(const std::vector<float>& spectrum, double binIndex) const
{
    if (spectrum.empty()) return 0;
    int n = static_cast<int>(spectrum.size());
    if (binIndex <= 0) return spectrum[0];
    if (binIndex >= n - 1) return spectrum[n - 1];
    
    int i0 = static_cast<int>(std::floor(binIndex));
    int i1 = std::min(i0 + 1, n - 1);
    double frac = binIndex - i0;
    
    return static_cast<float>(spectrum[i0] * (1.0 - frac) + spectrum[i1] * frac);
}

// Note names for A440 12TET
static const char* NOTE_NAMES[] = {
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
};

// Get pitch direction based on cents deviation
LoiaconoRolling::PitchDirection LoiaconoRolling::getPitchDirection(double cents, double threshold)
{
    if (cents < -threshold) return PitchDirection::Flat;
    if (cents > threshold) return PitchDirection::Sharp;
    return PitchDirection::InTune;
}

// Detect root pitch using harmonic correlation
// Uses a simplified harmonic product spectrum approach adapted for log-spaced bins
LoiaconoRolling::PitchResult LoiaconoRolling::detectRootPitch(
    const std::vector<float>& spectrum, double minFreq, double maxFreq, double baseAFreq) const
{
    double referenceA = baseAFreq > 0 ? baseAFreq : baseAFreq_;  // Use provided or default
    PitchResult result;
    if (spectrum.empty() || numBins_ < 2) return result;
    
    // Find valid frequency range in bin indices
    double binMin = std::max(0.0, freqToBin(minFreq));
    double binMax = std::min(static_cast<double>(numBins_ - 1), freqToBin(maxFreq));
    if (binMin >= binMax) return result;
    
    // Find the peak bin for normalization
    float maxAmp = 0;
    for (float v : spectrum) maxAmp = std::max(maxAmp, v);
    if (maxAmp < 1e-10f) return result;
    
    // Search for fundamental using harmonic correlation
    // For each candidate fundamental, compute correlation with harmonics
    double bestBin = 0;
    double bestScore = 0;
    
    const int numHarmonics = 6;  // Check up to 6th harmonic
    const double harmonicWeights[] = {1.0, 0.8, 0.6, 0.5, 0.4, 0.35};  // Fundamentals weighted more
    
    // Step through candidate fundamentals in log space
    int numSteps = std::max(10, numBins_ / 4);  // Reasonable resolution
    for (int step = 0; step < numSteps; step++) {
        double t = step / static_cast<double>(numSteps - 1);
        double candidateBin = binMin + t * (binMax - binMin);
        double candidateFreq = binToFreqHz(candidateBin);
        
        // Compute harmonic correlation score
        double score = 0;
        double weightSum = 0;
        
        for (int h = 0; h < numHarmonics; h++) {
            double harmonicFreq = candidateFreq * (h + 1);
            if (harmonicFreq > freqs_[numBins_ - 1] * sampleRate_ * 0.95) break;
            
            double harmonicBin = freqToBin(harmonicFreq);
            float amp = interpolatedSpectrum(spectrum, harmonicBin) / maxAmp;  // Normalized
            
            // Weight by expected harmonic amplitude roll-off (higher harmonics usually weaker)
            score += amp * harmonicWeights[h];
            weightSum += harmonicWeights[h];
        }
        
        if (weightSum > 0) {
            score /= weightSum;  // Normalize by weights
            // Penalize very low frequencies (often noise)
            if (candidateFreq < 80) score *= 0.8;
        }
        
        if (score > bestScore) {
            bestScore = score;
            bestBin = candidateBin;
        }
    }
    
    if (bestScore < 0.05) return result;  // Too quiet/noisy
    
    // Refine using quadratic interpolation around the peak
    double refinedBin = bestBin;
    int iBest = static_cast<int>(std::round(bestBin));
    if (iBest > 0 && iBest < numBins_ - 1) {
        double y0 = 0, y1 = bestScore, y2 = 0;
        // Recompute neighbor scores for interpolation
        for (int offset : {-1, 1}) {
            double neighborBin = bestBin + offset * 0.5;
            double neighborFreq = binToFreqHz(neighborBin);
            double score = 0;
            for (int h = 0; h < numHarmonics; h++) {
                double harmonicFreq = neighborFreq * (h + 1);
                if (harmonicFreq > freqs_[numBins_ - 1] * sampleRate_ * 0.95) break;
                double harmonicBin = freqToBin(harmonicFreq);
                score += interpolatedSpectrum(spectrum, harmonicBin) / maxAmp * harmonicWeights[h];
            }
            if (offset == -1) y0 = score / numHarmonics;
            else y2 = score / numHarmonics;
        }
        // Parabolic interpolation: peak at x = i + (y0 - y2) / (2*(y0 - 2*y1 + y2))
        double denom = 2.0 * (y0 - 2.0 * y1 + y2);
        if (std::abs(denom) > 1e-10) {
            double shift = (y0 - y2) / denom;
            refinedBin = bestBin + std::clamp(shift, -0.5, 0.5);
        }
    }
    
    double detectedFreq = binToFreqHz(refinedBin);
    
    // Convert to MIDI note number (A4 = referenceA Hz = MIDI 69)
    // MIDI note = 69 + 12 * log2(freq / referenceA)
    double midiExact = 69.0 + 12.0 * std::log2(detectedFreq / referenceA);
    int midiNote = static_cast<int>(std::round(midiExact));
    midiNote = std::clamp(midiNote, 0, 127);
    
    // Calculate cents deviation
    double cents = (midiExact - std::round(midiExact)) * 100.0;
    
    // Get note name
    int noteIndex = ((midiNote % 12) + 12) % 12;  // Handle negative
    int octave = (midiNote / 12) - 1;
    static char noteNameBuf[16];
    std::snprintf(noteNameBuf, sizeof(noteNameBuf), "%s%d", NOTE_NAMES[noteIndex], octave);
    
    // Store result
    result.freqHz = detectedFreq;
    result.confidence = std::min(1.0, bestScore);
    result.midiNote = midiNote;
    result.cents = cents;
    result.noteName = noteNameBuf;
    
    return result;
}
