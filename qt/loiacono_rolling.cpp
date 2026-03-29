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

    // Build log-spaced frequency bins
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

        // Window = multiple periods of this frequency
        int wlen = static_cast<int>(multiple / fNorm);
        wlen = std::min(wlen, RING_SIZE);
        windowLens_[i] = wlen;
        norms_[i] = 1.0 / std::sqrt(static_cast<double>(wlen));
    }

    // Reset accumulators
    Tr_.assign(numBins, 0.0);
    Ti_.assign(numBins, 0.0);

    // Reset ring buffer
    ring_.assign(RING_SIZE, 0.0f);
    ringHead_ = 0;
    sampleCount_ = 0;
}

void LoiaconoRolling::processSample(float sample)
{
    // Store new sample in ring buffer
    ring_[ringHead_] = sample;

    // For each frequency bin: comb filter + resonator update
    for (int fi = 0; fi < numBins_; fi++) {
        double f = freqs_[fi];
        double norm = norms_[fi];
        int wlen = windowLens_[fi];

        // Resonator: add new sample's contribution
        //   Tr += x[n] * cos(2π·f·n) / sqrt(W)
        //   Ti -= x[n] * sin(2π·f·n) / sqrt(W)
        double angle = TWO_PI * f * static_cast<double>(sampleCount_);
        Tr_[fi] += sample * std::cos(angle) * norm;
        Ti_[fi] -= sample * std::sin(angle) * norm;

        // Comb filter: subtract oldest sample leaving the window
        //   The sample that entered W steps ago is at ring[(head - W + RING_SIZE) % RING_SIZE]
        //   Its phase was at (sampleCount_ - W)
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
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < count; i++) {
        processSample(samples[i]);
    }
}

void LoiaconoRolling::getSpectrum(std::vector<float>& out) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    out.resize(numBins_);
    for (int i = 0; i < numBins_; i++) {
        out[i] = static_cast<float>(std::sqrt(Tr_[i] * Tr_[i] + Ti_[i] * Ti_[i]));
    }
}
