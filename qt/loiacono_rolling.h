#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <chrono>
#include <thread>

class LoiaconoRolling {
public:
    enum class ComputeMode {
        SingleThread,
        MultiThread,
        GpuCompute,
    };

    LoiaconoRolling() = default;

    void configure(double sampleRate, double freqMin, double freqMax,
                   int numBins, int multiple);

    void processSample(float sample);
    void processChunk(const float* samples, int count);
    void getSpectrum(std::vector<float>& out) const;

    int numBins() const { return numBins_; }
    double binFreqHz(int i) const { return freqs_[i] * sampleRate_; }
    double sampleRate() const { return sampleRate_; }
    int multiple() const { return multiple_; }
    unsigned int cpuThreads() const { return workerCount_; }
    void setComputeMode(ComputeMode mode) { computeMode_ = mode; }
    ComputeMode computeMode() const { return computeMode_; }
    ComputeMode activeComputeMode() const;
    static const char* computeModeName(ComputeMode mode);
    bool gpuComputeAvailable() const { return false; }

    // Runtime stats (thread-safe reads)
    struct Stats {
        uint64_t totalSamples = 0;
        uint64_t totalChunks = 0;
        double uptimeSeconds = 0;
        double samplesPerSecond = 0;
        double avgChunkMicros = 0;   // average processChunk time
        double peakChunkMicros = 0;  // worst-case processChunk time
        double cpuLoadPercent = 0;   // fraction of audio deadline used
        int currentBins = 0;
        int currentMultiple = 0;
        double freqMin = 0;
        double freqMax = 0;
    };
    Stats getStats() const;

private:
    double sampleRate_ = 48000;
    int multiple_ = 40;
    int numBins_ = 0;

    std::vector<double> freqs_;
    std::vector<int>    windowLens_;
    std::vector<double> norms_;
    std::vector<double> Tr_, Ti_;

    static constexpr int RING_SIZE = 1 << 15;
    std::vector<float> ring_;
    int ringHead_ = 0;
    uint64_t sampleCount_ = 0;

    // Stats tracking
    using Clock = std::chrono::steady_clock;
    Clock::time_point startTime_ = Clock::now();
    uint64_t chunkCount_ = 0;
    double totalChunkMicros_ = 0;
    double peakChunkMicros_ = 0;
    uint64_t lastChunkSamples_ = 0;
    unsigned int workerCount_ = std::max(1u, std::thread::hardware_concurrency());
    ComputeMode computeMode_ = ComputeMode::MultiThread;

    mutable std::mutex mutex_;
};
