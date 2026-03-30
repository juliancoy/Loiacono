#include "loiacono_rolling.h"
#include <algorithm>

static constexpr double TWO_PI = 2.0 * M_PI;

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
        for (int i = 0; i < count; i++) {
            processSample(samples[i]);
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
