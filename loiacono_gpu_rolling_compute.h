#pragma once

#include <cstdint>
#include <memory>
#include <vector>

class LoiaconoGpuRollingCompute {
public:
    LoiaconoGpuRollingCompute();
    ~LoiaconoGpuRollingCompute();

    bool available() const;
    bool configure(int signalLength,
                   int maxChunkLength,
                   int numBins,
                   const std::vector<double>& freqs,
                   const std::vector<double>& norms,
                   const std::vector<int>& windowLens);
    bool processChunk(const float* newSamples,
                      const float* oldSamples,
                      int count,
                      std::uint64_t startSampleCount);
    bool spectrum(std::vector<float>& outSpectrum) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
