// Headless CPU consistency test - no Qt, no OpenGL, no GPU
// This test verifies single-threaded and multi-threaded CPU implementations
// produce identical results

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <thread>
#include <mutex>

// Test parameters
constexpr double SAMPLE_RATE = 48000.0;
constexpr int SIGNAL_LENGTH = 1 << 12;  // 4096 samples
constexpr int MULTIPLE = 40;
constexpr double FREQ_MIN = 100.0;   // Hz
constexpr double FREQ_MAX = 3000.0;  // Hz
constexpr int NUM_BINS = 100;
constexpr double TWO_PI = 2.0 * M_PI;
constexpr int RING_SIZE = 32768;

// Tolerance for comparisons
constexpr double ABS_TOLERANCE = 1e-3;

// ============================================
// Direct Goertzel implementation
// ============================================
struct GoertzelBin {
    double fNorm;      // Normalized frequency (f/sampleRate)
    double norm;       // Amplitude normalization factor
    int windowLen;     // Window length
    double Tr = 0.0;   // Real part
    double Ti = 0.0;   // Imaginary part
};

std::vector<GoertzelBin> configureBins(double sampleRate, double freqMin, double freqMax,
                                        int numBins, int multiple) {
    std::vector<GoertzelBin> bins;
    bins.reserve(numBins);
    
    double logMin = std::log(freqMin);
    double logMax = std::log(freqMax);
    double logStep = (logMax - logMin) / numBins;
    
    for (int i = 0; i < numBins; i++) {
        GoertzelBin bin;
        double fHz = std::exp(logMin + i * logStep);
        bin.fNorm = fHz / sampleRate;
        bin.windowLen = std::min(static_cast<int>(multiple / bin.fNorm), RING_SIZE);
        // Goertzel amplitude scaling: 1/sqrt(N) to normalize for window size
        bin.norm = 1.0 / std::sqrt(static_cast<double>(bin.windowLen));
        bins.push_back(bin);
    }
    return bins;
}

// ============================================
// Single-threaded processing
// ============================================
class SingleThreadGoertzel {
public:
    void configure(double sampleRate, double freqMin, double freqMax,
                   int numBins, int multiple) {
        bins_ = configureBins(sampleRate, freqMin, freqMax, numBins, multiple);
        ring_.assign(RING_SIZE, 0.0f);
        ringHead_ = 0;
        sampleCount_ = 0;
        for (auto& bin : bins_) {
            bin.Tr = 0.0;
            bin.Ti = 0.0;
        }
    }
    
    void processChunk(const float* samples, int count) {
        for (int i = 0; i < count; i++) {
            float sample = samples[i];
            ring_[ringHead_] = sample;
            
            uint64_t n = sampleCount_;
            for (auto& bin : bins_) {
                // Rolling Goertzel recurrence
                double angle = TWO_PI * bin.fNorm * static_cast<double>(n);
                bin.Tr += sample * std::cos(angle) * bin.norm;
                bin.Ti -= sample * std::sin(angle) * bin.norm;
                
                if (n >= static_cast<uint64_t>(bin.windowLen)) {
                    int oldIdx = (ringHead_ - bin.windowLen + RING_SIZE) % RING_SIZE;
                    float oldSample = ring_[oldIdx];
                    double oldAngle = TWO_PI * bin.fNorm * static_cast<double>(n - bin.windowLen);
                    bin.Tr -= oldSample * std::cos(oldAngle) * bin.norm;
                    bin.Ti += oldSample * std::sin(oldAngle) * bin.norm;
                }
            }
            
            ringHead_ = (ringHead_ + 1) % RING_SIZE;
            sampleCount_++;
        }
    }
    
    std::vector<float> getSpectrum() const {
        std::vector<float> spectrum;
        spectrum.reserve(bins_.size());
        for (const auto& bin : bins_) {
            spectrum.push_back(static_cast<float>(std::sqrt(bin.Tr * bin.Tr + bin.Ti * bin.Ti)));
        }
        return spectrum;
    }
    
private:
    std::vector<GoertzelBin> bins_;
    std::vector<float> ring_;
    int ringHead_ = 0;
    uint64_t sampleCount_ = 0;
};

// ============================================
// Multi-threaded processing
// ============================================
class MultiThreadGoertzel {
public:
    void configure(double sampleRate, double freqMin, double freqMax,
                   int numBins, int multiple) {
        bins_ = configureBins(sampleRate, freqMin, freqMax, numBins, multiple);
        ring_.assign(RING_SIZE, 0.0f);
        ringHead_ = 0;
        sampleCount_ = 0;
    }
    
    void processChunk(const float* samples, int count) {
        // We need the ring buffer state including where new samples will be written
        // Build a view of the ring buffer that includes new samples at correct positions
        std::vector<float> ringView = ring_;
        for (int i = 0; i < count; i++) {
            ringView[(ringHead_ + i) % RING_SIZE] = samples[i];
        }
        
        int ringHead = ringHead_;
        uint64_t startSampleCount = sampleCount_;
        auto bins = bins_;  // Copy bin configs (not state)
        
        unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());
        size_t binsPerThread = (bins.size() + numThreads - 1) / numThreads;
        
        std::vector<std::thread> threads;
        std::mutex resultMutex;
        std::vector<std::pair<size_t, std::vector<GoertzelBin>>> results;
        
        for (unsigned int t = 0; t < numThreads; t++) {
            size_t startBin = t * binsPerThread;
            size_t endBin = std::min((t + 1) * binsPerThread, bins.size());
            if (startBin >= endBin) continue;
            
            threads.emplace_back([&, ringView, ringHead, startSampleCount, startBin, endBin]() {
                // Each thread gets a copy of the bin states it needs to process
                std::vector<GoertzelBin> localBins(bins.begin() + startBin, bins.begin() + endBin);
                
                for (int i = 0; i < count; i++) {
                    float sample = ringView[(ringHead + i) % RING_SIZE];
                    uint64_t n = startSampleCount + i;
                    int idx = (ringHead + i) % RING_SIZE;
                    
                    for (auto& bin : localBins) {
                        double angle = TWO_PI * bin.fNorm * static_cast<double>(n);
                        bin.Tr += sample * std::cos(angle) * bin.norm;
                        bin.Ti -= sample * std::sin(angle) * bin.norm;
                        
                        if (n >= static_cast<uint64_t>(bin.windowLen)) {
                            int oldIdx = (idx - bin.windowLen + RING_SIZE) % RING_SIZE;
                            float oldSample = ringView[oldIdx];
                            double oldAngle = TWO_PI * bin.fNorm * static_cast<double>(n - bin.windowLen);
                            bin.Tr -= oldSample * std::cos(oldAngle) * bin.norm;
                            bin.Ti += oldSample * std::sin(oldAngle) * bin.norm;
                        }
                    }
                }
                
                std::lock_guard<std::mutex> lock(resultMutex);
                results.emplace_back(startBin, std::move(localBins));
            });
        }
        
        for (auto& t : threads) {
            t.join();
        }
        
        // Merge results
        for (auto& [startIdx, localBins] : results) {
            for (size_t i = 0; i < localBins.size(); i++) {
                bins_[startIdx + i] = localBins[i];
            }
        }
        
        // Update ring buffer and head
        for (int i = 0; i < count; i++) {
            ring_[(ringHead_ + i) % RING_SIZE] = samples[i];
        }
        ringHead_ = (ringHead_ + count) % RING_SIZE;
        sampleCount_ += count;
    }
    
    std::vector<float> getSpectrum() const {
        std::vector<float> spectrum;
        spectrum.reserve(bins_.size());
        for (const auto& bin : bins_) {
            spectrum.push_back(static_cast<float>(std::sqrt(bin.Tr * bin.Tr + bin.Ti * bin.Ti)));
        }
        return spectrum;
    }
    
private:
    std::vector<GoertzelBin> bins_;
    std::vector<float> ring_;
    int ringHead_ = 0;
    uint64_t sampleCount_ = 0;
};

// ============================================
// Test utilities
// ============================================
std::vector<float> generate_sine_wave(double freq_hz, int length) {
    std::vector<float> signal(length);
    for (int i = 0; i < length; ++i) {
        double t = i / SAMPLE_RATE;
        signal[i] = std::sin(2.0 * M_PI * freq_hz * t);
    }
    return signal;
}

std::vector<float> generate_harmonics(double fundamental_hz, int length, int num_harmonics) {
    std::vector<float> signal(length, 0.0f);
    for (int h = 1; h <= num_harmonics; ++h) {
        double freq_hz = fundamental_hz * h;
        for (int i = 0; i < length; ++i) {
            double t = i / SAMPLE_RATE;
            signal[i] += std::sin(2.0 * M_PI * freq_hz * t) / h;
        }
    }
    return signal;
}

std::vector<float> generate_white_noise(int length) {
    std::vector<float> signal(length);
    std::srand(42);
    for (int i = 0; i < length; ++i) {
        signal[i] = 2.0f * (std::rand() / float(RAND_MAX)) - 1.0f;
    }
    return signal;
}

double compute_max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, static_cast<double>(std::abs(a[i] - b[i])));
    }
    return max_diff;
}

// ============================================
// Main test
// ============================================
int main() {
    std::cout << "Loiacono CPU Consistency Test (Headless)" << std::endl;
    std::cout << "=========================================" << std::endl;
    std::cout << "Testing single-threaded vs multi-threaded implementations" << std::endl;
    
    // Generate test signals
    std::cout << "\nGenerating test signals..." << std::endl;
    auto sine_440 = generate_sine_wave(440.0, SIGNAL_LENGTH);
    auto harmonics = generate_harmonics(220.0, SIGNAL_LENGTH, 5);
    auto white_noise = generate_white_noise(SIGNAL_LENGTH);
    
    std::vector<std::pair<std::string, std::vector<float>>> test_signals = {
        {"sine_440", sine_440},
        {"harmonics", harmonics},
        {"white_noise", white_noise}
    };
    
    bool all_passed = true;
    
    for (const auto& [name, signal] : test_signals) {
        std::cout << "\nTesting signal: " << name << std::endl;
        
        // Single-threaded
        SingleThreadGoertzel single;
        single.configure(SAMPLE_RATE, FREQ_MIN, FREQ_MAX, NUM_BINS, MULTIPLE);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        single.processChunk(signal.data(), signal.size());
        auto t1 = std::chrono::high_resolution_clock::now();
        auto spec_single = single.getSpectrum();
        double time_single = std::chrono::duration<double>(t1 - t0).count();
        
        // Multi-threaded
        MultiThreadGoertzel multi;
        multi.configure(SAMPLE_RATE, FREQ_MIN, FREQ_MAX, NUM_BINS, MULTIPLE);
        
        t0 = std::chrono::high_resolution_clock::now();
        multi.processChunk(signal.data(), signal.size());
        t1 = std::chrono::high_resolution_clock::now();
        auto spec_multi = multi.getSpectrum();
        double time_multi = std::chrono::duration<double>(t1 - t0).count();
        
        // Compare
        double max_diff = compute_max_abs_diff(spec_single, spec_multi);
        
        std::cout << "  Single-threaded: " << std::fixed << std::setprecision(4) 
                  << time_single * 1000 << " ms" << std::endl;
        std::cout << "  Multi-threaded:  " << time_multi * 1000 << " ms" << std::endl;
        std::cout << "  Max difference:  " << std::scientific << max_diff << std::endl;
        
        if (max_diff > ABS_TOLERANCE) {
            std::cout << "  FAILED: difference exceeds tolerance " << ABS_TOLERANCE << std::endl;
            all_passed = false;
        } else {
            std::cout << "  PASSED" << std::endl;
        }
        
        if (time_multi > 0) {
            std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                      << time_single / time_multi << "x" << std::endl;
        }
    }
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    if (all_passed) {
        std::cout << "ALL TESTS PASSED! ✓" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED! ✗" << std::endl;
        return 1;
    }
}
