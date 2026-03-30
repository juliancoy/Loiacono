#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <QApplication>

#include "../loiacono_rolling.h"
#include "../loiacono_parallel.h"

// Test parameters
constexpr double SAMPLE_RATE = 48000.0;
constexpr int SIGNAL_LENGTH = 1 << 12;  // 4096 samples
constexpr int MULTIPLE = 40;
constexpr double FREQ_MIN = 100.0;   // Hz
constexpr double FREQ_MAX = 3000.0;  // Hz
constexpr int NUM_BINS = 100;

// Tolerance for comparisons
constexpr double ABS_TOLERANCE = 1e-3;
constexpr double REL_TOLERANCE = 1e-2;

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
    
    std::cout << "    " << name << ": " << spectrum_out.size() 
              << " bins, time: " << std::fixed << std::setprecision(4) 
              << elapsed_time << "s" << std::endl;
}

bool compare_implementations(const std::string& signal_name,
                             const std::vector<float>& spectrum_single,
                             const std::vector<float>& spectrum_multi,
                             const std::vector<float>& spectrum_gpu,
                             double time_single,
                             double time_multi,
                             double time_gpu) {
    
    std::cout << "\n  Signal: " << signal_name << std::endl;
    
    bool all_passed = true;
    
    // Compare single-threaded vs multi-threaded
    if (!spectrum_single.empty() && !spectrum_multi.empty()) {
        double max_diff = compute_max_abs_diff(spectrum_single, spectrum_multi);
        double mean_diff = compute_mean_abs_diff(spectrum_single, spectrum_multi);
        double max_rel_diff = compute_max_rel_diff(spectrum_single, spectrum_multi);
        
        std::cout << "    Single vs Multi-threaded:" << std::endl;
        std::cout << "      Max abs diff: " << std::scientific << std::setprecision(2) << max_diff << std::endl;
        std::cout << "      Mean abs diff: " << std::scientific << std::setprecision(2) << mean_diff << std::endl;
        std::cout << "      Max rel diff: " << std::scientific << std::setprecision(2) << max_rel_diff << std::endl;
        
        if (max_diff > ABS_TOLERANCE) {
            std::cout << "      WARNING: Large absolute difference (> " << ABS_TOLERANCE << ")" << std::endl;
            all_passed = false;
        }
        
        if (time_multi > 0) {
            std::cout << "      Speedup: " << std::fixed << std::setprecision(2) 
                      << time_single / time_multi << "x" << std::endl;
        }
    }
    
    // Compare single-threaded vs GPU (if available)
    if (!spectrum_single.empty() && !spectrum_gpu.empty()) {
        double max_diff = compute_max_abs_diff(spectrum_single, spectrum_gpu);
        double mean_diff = compute_mean_abs_diff(spectrum_single, spectrum_gpu);
        double max_rel_diff = compute_max_rel_diff(spectrum_single, spectrum_gpu);
        
        std::cout << "    Single-threaded vs GPU:" << std::endl;
        std::cout << "      Max abs diff: " << std::scientific << std::setprecision(2) << max_diff << std::endl;
        std::cout << "      Mean abs diff: " << std::scientific << std::setprecision(2) << mean_diff << std::endl;
        std::cout << "      Max rel diff: " << std::scientific << std::setprecision(2) << max_rel_diff << std::endl;
        
        if (max_diff > ABS_TOLERANCE) {
            std::cout << "      WARNING: Large absolute difference (> " << ABS_TOLERANCE << ")" << std::endl;
            all_passed = false;
        }
        
        if (time_gpu > 0) {
            std::cout << "      Speedup: " << std::fixed << std::setprecision(2) 
                      << time_single / time_gpu << "x" << std::endl;
        }
    }
    
    return all_passed;
}

int main() {
    std::cout << "Loiacono C++ Implementation Consistency Tests" << std::endl;
    std::cout << "==============================================" << std::endl;
    
    // Generate test signals
    std::cout << "Generating test signals..." << std::endl;
    
    auto sine_440 = generate_sine_wave(440.0, SIGNAL_LENGTH);
    auto harmonics = generate_harmonics(440.0, SIGNAL_LENGTH, 5);
    auto white_noise = generate_white_noise(SIGNAL_LENGTH);
    auto chirp = generate_chirp(100.0, 2000.0, SIGNAL_LENGTH);
    
    std::vector<std::pair<std::string, std::vector<float>>> test_signals = {
        {"sine_440", sine_440},
        {"harmonics", harmonics},
        {"white_noise", white_noise},
        {"chirp", chirp}
    };
    
    std::cout << "Generated " << test_signals.size() << " test signals" << std::endl;
    
    bool all_tests_passed = true;
    
    for (const auto& [signal_name, signal] : test_signals) {
        std::cout << "\nTesting signal: " << signal_name << std::endl;
        
        std::vector<float> spectrum_single, spectrum_multi, spectrum_gpu;
        double time_single = 0.0, time_multi = 0.0, time_gpu = 0.0;
        
        // Test single-threaded implementation
        test_implementation("Single-threaded", 
                           LoiaconoRolling::ComputeMode::SingleThread,
                           signal, spectrum_single, time_single);
        
        // Test multi-threaded implementation
        test_implementation("Multi-threaded",
                           LoiaconoRolling::ComputeMode::MultiThread,
                           signal, spectrum_multi, time_multi);
        
        // Test GPU implementation (if available)
        LoiaconoRolling loiacono_gpu_test;
        loiacono_gpu_test.setComputeMode(LoiaconoRolling::ComputeMode::GpuCompute);
        loiacono_gpu_test.configure(SAMPLE_RATE, FREQ_MIN, FREQ_MAX, NUM_BINS, MULTIPLE);
        
        if (loiacono_gpu_test.gpuComputeAvailable()) {
            test_implementation("GPU",
                               LoiaconoRolling::ComputeMode::GpuCompute,
                               signal, spectrum_gpu, time_gpu);
        } else {
            std::cout << "    GPU: Not available" << std::endl;
        }
        
        // Compare implementations
        bool test_passed = compare_implementations(signal_name,
                                                  spectrum_single,
                                                  spectrum_multi,
                                                  spectrum_gpu,
                                                  time_single,
                                                  time_multi,
                                                  time_gpu);
        
        if (!test_passed) {
            all_tests_passed = false;
        }
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_tests_passed) {
        std::cout << "ALL TESTS PASSED! ✓" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED! ✗" << std::endl;
        return 1;
    }
}