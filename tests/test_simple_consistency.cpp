#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <iomanip>

#include "../loiacono_rolling.h"
#include "../loiacono_parallel.h"

// Test parameters
constexpr double SAMPLE_RATE = 48000.0;
constexpr int SIGNAL_LENGTH = 1 << 12;  // 4096 samples
constexpr int MULTIPLE = 40;
constexpr double FREQ_MIN = 100.0;   // Hz
constexpr double FREQ_MAX = 3000.0;  // Hz
constexpr int NUM_BINS = 100;
constexpr double DEFAULT_LEAKINESS = 0.99995;
constexpr double TEST_LEAKINESS = 0.995;
constexpr int SATURATION_SIGNAL_LENGTH = 1 << 16;
constexpr int SPEED_SIGNAL_LENGTH = 1 << 15;
constexpr int SPEED_BINS = 400;
constexpr int SPEED_REPEATS = 3;
constexpr double MAX_LEAKY_SLOWDOWN = 3.0;
constexpr double MAX_TAPERED_SLOWDOWN = 12.0;

// Tolerance for comparisons
constexpr double ABS_TOLERANCE = 1e-3;

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

void process_in_chunks(LoiaconoRolling& loiacono,
                       const std::vector<float>& signal,
                       const std::vector<int>& chunkPattern)
{
    size_t offset = 0;
    size_t patternIdx = 0;
    while (offset < signal.size()) {
        int chunk = chunkPattern[patternIdx % chunkPattern.size()];
        int count = std::min<int>(chunk, static_cast<int>(signal.size() - offset));
        loiacono.processChunk(signal.data() + offset, count);
        offset += static_cast<size_t>(count);
        patternIdx++;
    }
}

bool run_ring_accuracy_test()
{
    constexpr double satFreqMin = 20.0;
    constexpr double satFreqMax = 220.0;
    constexpr int satBins = 64;
    constexpr int satMultiple = 240;

    std::cout << "\nRing buffer sample-accuracy test" << std::endl;
    std::cout << "  Config: " << satFreqMin << "-" << satFreqMax
              << " Hz, bins=" << satBins << ", multiple=" << satMultiple
              << ", samples=" << SATURATION_SIGNAL_LENGTH << std::endl;

    auto longSine = generate_sine_wave(55.0, SATURATION_SIGNAL_LENGTH);
    auto longHarmonics = generate_harmonics(41.25, SATURATION_SIGNAL_LENGTH, 4);
    std::vector<std::pair<std::string, std::vector<float>>> signals = {
        {"long_sine_55", longSine},
        {"long_harmonics_41", longHarmonics},
    };
    const std::vector<std::vector<int>> chunkPatterns = {
        {1},
        {17, 31, 257, 509},
        {256},
        {511, 64, 13, 1024},
    };

    bool passed = true;
    for (const auto& [name, signal] : signals) {
        LoiaconoRolling reference;
        reference.setComputeMode(LoiaconoRolling::ComputeMode::SingleThread);
        reference.configure(SAMPLE_RATE, satFreqMin, satFreqMax, satBins, satMultiple);
        for (float sample : signal) {
            reference.processSample(sample);
        }
        std::vector<float> refSpectrum;
        reference.getSpectrum(refSpectrum);

        std::cout << "  Signal: " << name << std::endl;
        for (const auto& pattern : chunkPatterns) {
            LoiaconoRolling single;
            single.setComputeMode(LoiaconoRolling::ComputeMode::SingleThread);
            single.configure(SAMPLE_RATE, satFreqMin, satFreqMax, satBins, satMultiple);
            process_in_chunks(single, signal, pattern);
            std::vector<float> singleSpectrum;
            single.getSpectrum(singleSpectrum);

            LoiaconoRolling multi;
            multi.setComputeMode(LoiaconoRolling::ComputeMode::MultiThread);
            multi.configure(SAMPLE_RATE, satFreqMin, satFreqMax, satBins, satMultiple);
            process_in_chunks(multi, signal, pattern);
            std::vector<float> multiSpectrum;
            multi.getSpectrum(multiSpectrum);

            double singleDiff = compute_max_abs_diff(refSpectrum, singleSpectrum);
            double multiDiff = compute_max_abs_diff(refSpectrum, multiSpectrum);
            std::cout << "    Chunk pattern";
            for (int chunk : pattern) std::cout << " " << chunk;
            std::cout << ": single=" << std::scientific << std::setprecision(2) << singleDiff
                      << ", multi=" << multiDiff;
            if (singleDiff > ABS_TOLERANCE || multiDiff > ABS_TOLERANCE) {
                std::cout << " [FAIL]";
                passed = false;
            } else {
                std::cout << " [PASS]";
            }
            std::cout << std::endl;
        }
    }

    return passed;
}

void test_implementation(const std::string& name, 
                         LoiaconoRolling::ComputeMode mode,
                         LoiaconoRolling::WindowMode weightingMode,
                         LoiaconoRolling::NormalizationMode normalizationMode,
                         double leakiness,
                         const std::vector<float>& signal,
                         std::vector<float>& spectrum_out,
                         double& elapsed_time) {
    
    LoiaconoRolling loiacono;
    loiacono.setComputeMode(mode);
    loiacono.setWindowMode(weightingMode);
    loiacono.setNormalizationMode(normalizationMode);
    loiacono.setLeakiness(leakiness);
    loiacono.configure(SAMPLE_RATE, FREQ_MIN, FREQ_MAX, NUM_BINS, MULTIPLE);
    
    auto start = std::chrono::high_resolution_clock::now();
    loiacono.processChunk(signal.data(), signal.size());
    loiacono.getSpectrum(spectrum_out);
    auto end = std::chrono::high_resolution_clock::now();
    
    elapsed_time = std::chrono::duration<double>(end - start).count();
    
    std::cout << "    " << name << ": " << spectrum_out.size() 
              << " bins, time: " << std::fixed << std::setprecision(4) 
              << elapsed_time << "s" << std::endl;
}

double benchmark_full_cycle(LoiaconoRolling::ComputeMode mode,
                            LoiaconoRolling::WindowMode windowMode,
                            LoiaconoRolling::NormalizationMode normalizationMode,
                            double leakiness,
                            const std::vector<float>& signal,
                            int numBins,
                            int repeats)
{
    std::vector<double> timings;
    timings.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        LoiaconoRolling loiacono;
        loiacono.setComputeMode(mode);
        loiacono.setWindowMode(windowMode);
        loiacono.setNormalizationMode(normalizationMode);
        loiacono.setLeakiness(leakiness);
        loiacono.configure(SAMPLE_RATE, FREQ_MIN, FREQ_MAX, numBins, MULTIPLE);

        std::vector<float> spectrum;
        auto start = std::chrono::high_resolution_clock::now();
        loiacono.processChunk(signal.data(), static_cast<int>(signal.size()));
        loiacono.getSpectrum(spectrum);
        auto end = std::chrono::high_resolution_clock::now();
        timings.push_back(std::chrono::duration<double>(end - start).count());
    }

    std::sort(timings.begin(), timings.end());
    return timings[timings.size() / 2];
}

bool run_speed_battery()
{
    std::cout << "\nSpeed battery" << std::endl;
    std::cout << "  Config: samples=" << SPEED_SIGNAL_LENGTH
              << ", bins=" << SPEED_BINS
              << ", repeats=" << SPEED_REPEATS << std::endl;

    auto signal = generate_white_noise(SPEED_SIGNAL_LENGTH);
    const double baseline = benchmark_full_cycle(
        LoiaconoRolling::ComputeMode::MultiThread,
        LoiaconoRolling::WindowMode::RectangularWindow,
        LoiaconoRolling::NormalizationMode::Energy,
        DEFAULT_LEAKINESS,
        signal,
        SPEED_BINS,
        SPEED_REPEATS);
    if (baseline <= 0.0) {
        std::cout << "  Baseline timing invalid [FAIL]" << std::endl;
        return false;
    }

    struct SpeedCase {
        const char* label;
        LoiaconoRolling::WindowMode windowMode;
        LoiaconoRolling::NormalizationMode normalizationMode;
        double leakiness;
        double maxSlowdown;
    };
    const std::vector<SpeedCase> cases = {
        {"rectangular + energy", LoiaconoRolling::WindowMode::RectangularWindow, LoiaconoRolling::NormalizationMode::Energy, DEFAULT_LEAKINESS, 1.25},
        {"leaky + energy", LoiaconoRolling::WindowMode::LeakyWindow, LoiaconoRolling::NormalizationMode::Energy, TEST_LEAKINESS, MAX_LEAKY_SLOWDOWN},
        {"hann + coherent", LoiaconoRolling::WindowMode::HannWindow, LoiaconoRolling::NormalizationMode::CoherentAmplitude, DEFAULT_LEAKINESS, MAX_TAPERED_SLOWDOWN},
        {"hamming + coherent", LoiaconoRolling::WindowMode::HammingWindow, LoiaconoRolling::NormalizationMode::CoherentAmplitude, DEFAULT_LEAKINESS, MAX_TAPERED_SLOWDOWN},
        {"blackman + energy", LoiaconoRolling::WindowMode::BlackmanWindow, LoiaconoRolling::NormalizationMode::Energy, DEFAULT_LEAKINESS, MAX_TAPERED_SLOWDOWN},
        {"blackman-harris + energy", LoiaconoRolling::WindowMode::BlackmanHarrisWindow, LoiaconoRolling::NormalizationMode::Energy, DEFAULT_LEAKINESS, MAX_TAPERED_SLOWDOWN},
    };

    bool passed = true;
    std::cout << "  Baseline multi-thread rectangular full-cycle: "
              << std::fixed << std::setprecision(4) << baseline << "s" << std::endl;
    for (const auto& speedCase : cases) {
        const double elapsed = benchmark_full_cycle(
            LoiaconoRolling::ComputeMode::MultiThread,
            speedCase.windowMode,
            speedCase.normalizationMode,
            speedCase.leakiness,
            signal,
            SPEED_BINS,
            SPEED_REPEATS);
        const double slowdown = elapsed / baseline;
        std::cout << "  " << speedCase.label << ": "
                  << std::fixed << std::setprecision(4) << elapsed << "s"
                  << " (" << std::setprecision(2) << slowdown << "x baseline)";
        if (slowdown > speedCase.maxSlowdown) {
            std::cout << " [FAIL > " << speedCase.maxSlowdown << "x]";
            passed = false;
        } else {
            std::cout << " [PASS]";
        }
        std::cout << std::endl;
    }

    return passed;
}

int main() {
    std::cout << "Loiacono C++ Implementation Consistency Tests" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Testing single-threaded vs multi-threaded CPU implementations" << std::endl;
    
    // Generate test signals
    std::cout << "Generating test signals..." << std::endl;
    
    auto sine_440 = generate_sine_wave(440.0, SIGNAL_LENGTH);
    auto harmonics = generate_harmonics(440.0, SIGNAL_LENGTH, 5);
    auto white_noise = generate_white_noise(SIGNAL_LENGTH);
    
    std::vector<std::pair<std::string, std::vector<float>>> test_signals = {
        {"sine_440", sine_440},
        {"harmonics", harmonics},
        {"white_noise", white_noise}
    };
    
    std::cout << "Generated " << test_signals.size() << " test signals" << std::endl;
    
    bool all_tests_passed = true;
    struct AnalysisCase {
        const char* label;
        LoiaconoRolling::WindowMode weightingMode;
        LoiaconoRolling::NormalizationMode normalizationMode;
        double leakiness;
    };
    const std::vector<AnalysisCase> analysis_cases = {
        {"rectangular + energy", LoiaconoRolling::WindowMode::RectangularWindow, LoiaconoRolling::NormalizationMode::Energy, DEFAULT_LEAKINESS},
        {"hann + coherent", LoiaconoRolling::WindowMode::HannWindow, LoiaconoRolling::NormalizationMode::CoherentAmplitude, DEFAULT_LEAKINESS},
        {"hamming + coherent", LoiaconoRolling::WindowMode::HammingWindow, LoiaconoRolling::NormalizationMode::CoherentAmplitude, DEFAULT_LEAKINESS},
        {"blackman + energy", LoiaconoRolling::WindowMode::BlackmanWindow, LoiaconoRolling::NormalizationMode::Energy, DEFAULT_LEAKINESS},
        {"blackman-harris + energy", LoiaconoRolling::WindowMode::BlackmanHarrisWindow, LoiaconoRolling::NormalizationMode::Energy, DEFAULT_LEAKINESS},
        {"leaky + energy", LoiaconoRolling::WindowMode::LeakyWindow, LoiaconoRolling::NormalizationMode::Energy, TEST_LEAKINESS},
    };
    
    for (const auto& analysis_case : analysis_cases) {
        std::cout << "\nAnalysis case: " << analysis_case.label
                  << " | leakiness=" << analysis_case.leakiness << std::endl;

        for (const auto& [signal_name, signal] : test_signals) {
            std::cout << "\nTesting signal: " << signal_name << std::endl;
            
            std::vector<float> spectrum_single, spectrum_multi;
            double time_single = 0.0, time_multi = 0.0;
            
            test_implementation("Single-threaded", 
                               LoiaconoRolling::ComputeMode::SingleThread,
                               analysis_case.weightingMode,
                               analysis_case.normalizationMode,
                               analysis_case.leakiness,
                               signal, spectrum_single, time_single);
            
            test_implementation("Multi-threaded",
                               LoiaconoRolling::ComputeMode::MultiThread,
                               analysis_case.weightingMode,
                               analysis_case.normalizationMode,
                               analysis_case.leakiness,
                               signal, spectrum_multi, time_multi);
            
            std::cout << "\n  Comparing single-threaded vs multi-threaded:" << std::endl;
            
            if (spectrum_single.size() != spectrum_multi.size()) {
                std::cout << "    ERROR: Dimension mismatch!" << std::endl;
                std::cout << "      Single-threaded: " << spectrum_single.size() << " bins" << std::endl;
                std::cout << "      Multi-threaded: " << spectrum_multi.size() << " bins" << std::endl;
                all_tests_passed = false;
                continue;
            }
            
            double max_diff = compute_max_abs_diff(spectrum_single, spectrum_multi);
            double mean_diff = compute_mean_abs_diff(spectrum_single, spectrum_multi);
            
            std::cout << "    Max absolute difference: " << std::scientific << std::setprecision(2) << max_diff << std::endl;
            std::cout << "    Mean absolute difference: " << std::scientific << std::setprecision(2) << mean_diff << std::endl;
            
            if (max_diff > ABS_TOLERANCE) {
                std::cout << "    WARNING: Large absolute difference (> " << ABS_TOLERANCE << ")" << std::endl;
                all_tests_passed = false;
            }
            
            if (time_multi > 0 && time_single > 0) {
                std::cout << "    Speedup: " << std::fixed << std::setprecision(2) 
                          << time_single / time_multi << "x" << std::endl;
            }
        }
    }

    if (!run_ring_accuracy_test()) {
        all_tests_passed = false;
    }
    if (!run_speed_battery()) {
        all_tests_passed = false;
    }
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    if (all_tests_passed) {
        std::cout << "ALL TESTS PASSED! ✓" << std::endl;
        std::cout << "Single-threaded and multi-threaded implementations produce consistent results." << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED! ✗" << std::endl;
        std::cout << "There are inconsistencies between implementations." << std::endl;
        return 1;
    }
}
