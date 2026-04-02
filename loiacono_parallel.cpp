#include "loiacono_parallel.h"

#include <algorithm>
#include <cmath>
#include <future>

namespace {
constexpr double TWO_PI = 2.0 * M_PI;

int overwrittenIndexForRingPos(int ringPos,
                               int startRingHead,
                               int ringSize,
                               int overwrittenCount)
{
    int rel = ringPos - startRingHead;
    if (rel < 0) rel += ringSize;
    return (rel >= 0 && rel < overwrittenCount) ? rel : -1;
}
}

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
    double leakiness)
{
    const int ringSize = static_cast<int>(ring.size());
    if (ringSize <= 0) return;

    auto processBinRange = [&](int begin, int end) {
        for (int fi = begin; fi < end; fi++) {
            double trValue = tr[fi];
            double tiValue = ti[fi];
            double f = freqs[fi];
            double norm = norms[fi];
            int wlen = windowLens[fi];

            for (int i = 0; i < count; i++) {
                // Apply leakiness per sample
                trValue *= leakiness;
                tiValue *= leakiness;

                uint64_t sampleIdx = startSampleCount + i;
                int writePos = (startRingHead + i) % ringSize;
                float sample = samples[i];

                double angle = TWO_PI * f * static_cast<double>(sampleIdx);
                trValue += sample * std::cos(angle) * norm;
                tiValue -= sample * std::sin(angle) * norm;

                if (sampleIdx >= static_cast<uint64_t>(wlen)) {
                    int oldIdx = (writePos - wlen + ringSize) % ringSize;
                    int overwrittenIdx = overwrittenIndexForRingPos(oldIdx,
                                                                    startRingHead,
                                                                    ringSize,
                                                                    static_cast<int>(overwrittenSamples.size()));
                    float oldSample = (overwrittenIdx > i)
                        ? overwrittenSamples[static_cast<size_t>(overwrittenIdx)]
                        : ring[oldIdx];
                    double oldAngle = TWO_PI * f * static_cast<double>(sampleIdx - wlen);
                    trValue -= oldSample * std::cos(oldAngle) * norm;
                    tiValue += oldSample * std::sin(oldAngle) * norm;
                }
            }

            tr[fi] = trValue;
            ti[fi] = tiValue;
        }
    };

    unsigned int threads = std::min<unsigned int>(workerCount, std::max(1, numBins));
    if (threads <= 1 || numBins < 64) {
        processBinRange(0, numBins);
        return;
    }

    std::vector<std::future<void>> jobs;
    jobs.reserve(threads);
    int binsPerThread = (numBins + static_cast<int>(threads) - 1) / static_cast<int>(threads);
    for (unsigned int t = 0; t < threads; t++) {
        int begin = static_cast<int>(t) * binsPerThread;
        int end = std::min(numBins, begin + binsPerThread);
        if (begin >= end) break;
        jobs.push_back(std::async(std::launch::async, processBinRange, begin, end));
    }
    for (auto& job : jobs) {
        job.get();
    }
}

}
