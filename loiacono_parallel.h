#pragma once

#include <cstdint>
#include <vector>

namespace loiacono {

void processBinsParallel(
    unsigned int workerCount,
    int numBins,
    int count,
    int startRingHead,
    uint64_t startSampleCount,
    const float* samples,
    const std::vector<float>& ring,
    const std::vector<double>& freqs,
    const std::vector<double>& norms,
    const std::vector<int>& windowLens,
    std::vector<double>& tr,
    std::vector<double>& ti,
    const std::vector<float>& overwrittenSamples,
    double leakiness);

}
