#include "loiacono_parallel.h"

#include <algorithm>
#include <cmath>
#include <future>

namespace {
constexpr double TWO_PI = 2.0 * M_PI;
constexpr int RING_SIZE = 1 << 15;
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
    double leakiness)
{
    auto processBinRange = [&](int begin, int end) {
        for (int fi = begin; fi < end; fi++) {
            double trValue = tr[fi];
            double tiValue = ti[fi];
            double f = freqs[fi];
            double norm = norms[fi];
            int wlen = windowLens[fi];

            // Apply leakiness once for the whole chunk (not per-sample)
            // leakiness^count approximates applying it count times
            double chunkLeak = std::pow(leakiness, count);
            trValue *= chunkLeak;
            tiValue *= chunkLeak;

            for (int i = 0; i < count; i++) {
                uint64_t sampleIdx = startSampleCount + i;
                int writePos = (startRingHead + i) % RING_SIZE;
                float sample = samples[i];

                double angle = TWO_PI * f * static_cast<double>(sampleIdx);
                trValue += sample * std::cos(angle) * norm;
                tiValue -= sample * std::sin(angle) * norm;

                if (sampleIdx >= static_cast<uint64_t>(wlen)) {
                    int oldIdx = (writePos - wlen + RING_SIZE) % RING_SIZE;
                    float oldSample = ring[oldIdx];
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
