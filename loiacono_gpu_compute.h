#pragma once

#include <memory>
#include <vector>

class LoiaconoGpuCompute {
public:
    LoiaconoGpuCompute();
    ~LoiaconoGpuCompute();

    bool available() const;
    bool configure(int signalLength,
                   int numBins,
                   const std::vector<double>& freqs,
                   const std::vector<double>& norms,
                   const std::vector<int>& windowLens,
                   int algorithmMode,
                   int windowMode,
                   int normalizationMode,
                   int fftLength);
    bool compute(const std::vector<float>& ring,
                 unsigned int offset,
                 unsigned int availableSamples,
                 float leakiness,
                 std::vector<float>& outSpectrum);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
