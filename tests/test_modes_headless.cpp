// GUI-less consistency test for Loiacono algorithm modes
// This test compares Single-thread and Multi-thread CPU modes
// GPU mode is disabled to avoid OpenGL context issues

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <cstring>

#include "../loiacono_rolling.h"
#include "../loiacono_parallel.h"

// Test parameters
constexpr double SAMPLE_RATE = 48000.0;
constexpr int SIGNAL_LENGTH = 1 << 14;  // 16384 samples (larger for better test)
constexpr int MULTIPLE = 40;
constexpr double FREQ_MIN = 100.0;   // Hz
constexpr double FREQ_MAX = 5000.0;  // Hz
constexpr int NUM_BINS = 200;

// Tolerance for comparisons
constexpr double ABS_TOLERANCE = 1e-3;
constexpr double REL_TOLERANCE = 0.05;  // 5% relative tolerance

// Use regular LoiaconoRolling - GPU is disabled via stubs above

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
    std::srand(42);  // Fixed seed for reproducibility
    for (int i = 0; i < length; ++i) {
        signal[i] = 2.0f * (std::rand() / float(RAND_MAX)) - 1.0f;
    }
    return signal;
}

std::vector<float> generate_chirp(double f0_hz, double f1_hz, int length) {
    std::vector<float> signal(length);
    double T = length / SAMPLE_RATE;
    for (int i = 0; i < length; ++i) {
        double t = i / SAMPLE_RATE;
        double phase = 2.0 * M_PI * (f0_hz * t + (f1_hz - f0_hz) * t * t / (2.0 * T));
        signal[i] = std::sin(phase);
    }
    return signal;
}

double compute_max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double max_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

double compute_mean_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    return sum / a.size();
}

double compute_max_rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        return std::numeric_limits<double>::infinity();
    }
    
    double max_rel_diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double denom = std::max(std::abs(a[i]), 1e-10f);
        double rel_diff = std::abs(a[i] - b[i]) / denom;
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }
    }
    return max_rel_diff;
}

void test_implementation(const std::string& name, 
                         LoiaconoRolling::ComputeMode mode,
                         const std::vector<float>& signal,
                         std::vector<float>& spectrum_out,
                         double& elapsed_time) {
    
    LoiaconoRolling loiacono;
    loiacono.setComputeMode(mode);
    loiacono.configure(SAMPLE_RATE, FREQ_MIN, FREQ_MAX, NUM_BINS, MULTIPLE);
    
    auto start = std::chrono::high_resolution_clock::now();
    loiacono.processChunk(signal.data(), signal.size());
    auto end = std::chrono::high_resolution_clock::now();
    
    loiacono.getSpectrum(spectrum_out);
    
    elapsed_time = std::chrono::duration<double>(end - start).count();
    
    std::cout << "  " << std::left << std::setw(15) << name << ": " 
              << spectrum_out.size() << " bins, "
              << std::fixed << std::setprecision(4) << elapsed_time << "s";
    
    // Show max amplitude
    float max_amp = 0;
    for (auto v : spectrum_out) max_amp = std::max(max_amp, v);
    std::cout << ", max_amp=" << std::fixed << std::setprecision(4) << max_amp;
    std::cout << std::endl;
}

bool compare_implementations(const std::string& signal_name,
                             const std::vector<float>& spectrum_single,
                             const std::vector<float>& spectrum_multi,
                             double time_single,
                             double time_multi) {
    
    std::cout << "\nComparison for: " << signal_name << std::endl;
    
    bool passed = true;
    
    if (!spectrum_single.empty() && !spectrum_multi.empty()) {
        double max_diff = compute_max_abs_diff(spectrum_single, spectrum_multi);
        double mean_diff = compute_mean_abs_diff(spectrum_single, spectrum_multi);
        double max_rel_diff = compute_max_rel_diff(spectrum_single, spectrum_multi);
        
        std::cout << "  Single vs Multi-threaded:" << std::endl;
        std::cout << "    Max abs diff: " << std::scientific << std::setprecision(2) << max_diff;
        if (max_diff > ABS_TOLERANCE) {
            std::cout << " [FAIL > " << ABS_TOLERANCE << "]";
            passed = false;
        } else {
            std::cout << " [PASS]";
        }
        std::cout << std::endl;
        
        std::cout << "    Mean abs diff: " << std::scientific << std::setprecision(2) << mean_diff << std::endl;
        
        std::cout << "    Max rel diff: " << std::scientific << std::setprecision(2) << max_rel_diff;
        if (max_rel_diff > REL_TOLERANCE) {
            std::cout << " [FAIL > " << REL_TOLERANCE << "]";
            passed = false;
        } else {
            std::cout << " [PASS]";
        }
        std::cout << std::endl;
        
        if (time_multi > 0) {
            std::cout << "    Speedup: " << std::fixed << std::setprecision(2) 
                      << time_single / time_multi << "x" << std::endl;
        }
    }
    
    return passed;
}

int main() {
    std::cout << "Loiacono Algorithm Consistency Test (Headless)" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  Sample rate: " << SAMPLE_RATE << " Hz" << std::endl;
    std::cout << "  Signal length: " << SIGNAL_LENGTH << " samples" << std::endl;
    std::cout << "  Freq range: " << FREQ_MIN << "-" << FREQ_MAX << " Hz" << std::endl;
    std::cout << "  Bins: " << NUM_BINS << std::endl;
    std::cout << "  Multiple: " << MULTIPLE << std::endl;
    std::cout << std::endl;
    
    // Generate test signals
    std::cout << "Generating test signals..." << std::endl;
    
    auto sine_440 = generate_sine_wave(440.0, SIGNAL_LENGTH);
    auto sine_1k = generate_sine_wave(1000.0, SIGNAL_LENGTH);
    auto harmonics = generate_harmonics(220.0, SIGNAL_LENGTH, 5);
    auto noise = generate_white_noise(SIGNAL_LENGTH);
    auto chirp = generate_chirp(100.0, 2000.0, SIGNAL_LENGTH);
    
    std::cout << "  Generated 5 test signals" << std::endl;
    std::cout << std::endl;
    
    // Test signals
    struct TestCase {
        std::string name;
        const std::vector<float>& signal;
    };
    
    std::vector<TestCase> test_cases = {
        {"sine_440", sine_440},
        {"sine_1k", sine_1k},
        {"harmonics", harmonics},
        {"noise", noise},
        {"chirp", chirp}
    };
    
    bool all_passed = true;
    
    for (const auto& test : test_cases) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing: " << test.name << std::endl;
        std::cout << "========================================" << std::endl;
        
        std::vector<float> spectrum_single, spectrum_multi;
        double time_single = 0, time_multi = 0;
        
        // Test single-threaded
        test_implementation("Single-thread", 
                           LoiaconoRolling::ComputeMode::SingleThread,
                           test.signal, spectrum_single, time_single);
        
        // Test multi-threaded
        test_implementation("Multi-thread", 
                           LoiaconoRolling::ComputeMode::MultiThread,
                           test.signal, spectrum_multi, time_multi);
        
        // Compare results
        if (!compare_implementations(test.name, spectrum_single, spectrum_multi,
                                     time_single, time_multi)) {
            all_passed = false;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL RESULT" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (all_passed) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
