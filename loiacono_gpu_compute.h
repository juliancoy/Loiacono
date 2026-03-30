#pragma once

#include <memory>
#include <vector>

class LoiaconoGpuCompute {
public:
    LoiaconoGpuCompute();
    ~LoiaconoGpuCompute();

    bool available() const;
    bool configure(int signalLength, int numBins, int multiple, const std::vector<double>& freqs);
    bool compute(const std::vector<float>& ring, unsigned int offset, std::vector<float>& outSpectrum);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
