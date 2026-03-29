#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <mutex>

// Rolling Loiacono Transform — O(numBins) per sample via comb filter + resonator
//
// For each frequency bin, maintains running real/imaginary accumulators.
// Each new audio sample:
//   1. ADD the new sample's twiddle-weighted contribution (resonator)
//   2. SUBTRACT the oldest sample leaving the window (comb filter: x[n] - x[n-W])
//
// The window length W = multiple * period = multiple / f_normalized
// varies per frequency bin (constant-Q analysis).

class LoiaconoRolling {
public:
    LoiaconoRolling() = default;

    // Configure with log-spaced frequency bins
    void configure(double sampleRate, double freqMin, double freqMax,
                   int numBins, int multiple);

    // Process a single audio sample — called from the audio thread
    void processSample(float sample);

    // Process a chunk of samples
    void processChunk(const float* samples, int count);

    // Read current spectrum magnitudes — called from the GUI thread
    // Thread-safe: copies from internal accumulators under lock
    void getSpectrum(std::vector<float>& out) const;

    int numBins() const { return numBins_; }
    double binFreqHz(int i) const { return freqs_[i] * sampleRate_; }
    double sampleRate() const { return sampleRate_; }

private:
    double sampleRate_ = 48000;
    int multiple_ = 40;
    int numBins_ = 0;

    // Per-bin parameters
    std::vector<double> freqs_;       // normalized frequencies (f / sr)
    std::vector<int>    windowLens_;   // window length per bin (samples)
    std::vector<double> norms_;        // 1/sqrt(windowLen) normalization

    // Per-bin running accumulators
    std::vector<double> Tr_, Ti_;     // real and imaginary sums

    // Ring buffer for the audio signal
    static constexpr int RING_SIZE = 1 << 15; // 32768
    std::vector<float> ring_;
    int ringHead_ = 0;
    uint64_t sampleCount_ = 0;

    mutable std::mutex mutex_;
};
