// Headless pitch-accuracy benchmark for tuning the root pitch detector.
// Focus: compare raw detector cents error vs UI-style smoothed cents error.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "../loiacono_rolling.h"

namespace {
constexpr double kSampleRate = 48000.0;
constexpr int kChunkFrames = 256;
constexpr double kDurationSeconds = 1.6;
constexpr double kWarmupSeconds = 0.6;
constexpr int kNumBins = 1024;
constexpr int kMultiple = 96;
constexpr double kFreqMin = 40.0;
constexpr double kFreqMax = 5000.0;
constexpr double kMinPitchHz = 50.0;
constexpr double kMaxPitchHz = 2000.0;
constexpr double kConfidenceThreshold = 0.3;
constexpr size_t kPitchHistorySize = 8;
constexpr double kTwoPi = 6.28318530717958647692;

enum class Waveform {
    Sine,
    Sawtooth,
    Square,
};

const char* waveformName(Waveform w)
{
    switch (w) {
    case Waveform::Sine: return "sine";
    case Waveform::Sawtooth: return "sawtooth";
    case Waveform::Square: return "square";
    }
    return "unknown";
}

std::vector<float> makeWaveChunk(Waveform waveform,
                                 double frequencyHz,
                                 double amplitude,
                                 uint64_t startSample,
                                 int frames)
{
    std::vector<float> out(static_cast<size_t>(frames), 0.0f);
    for (int i = 0; i < frames; ++i) {
        const double phase = std::fmod(((static_cast<double>(startSample + static_cast<uint64_t>(i)) * frequencyHz) / kSampleRate), 1.0);
        double sample = 0.0;
        switch (waveform) {
        case Waveform::Sine:
            sample = std::sin(phase * kTwoPi);
            break;
        case Waveform::Sawtooth:
            sample = (2.0 * phase) - 1.0;
            break;
        case Waveform::Square:
            sample = (phase < 0.5) ? 1.0 : -1.0;
            break;
        }
        out[static_cast<size_t>(i)] = static_cast<float>(sample * amplitude);
    }
    return out;
}

double centsError(double detectedHz, double targetHz)
{
    if (detectedHz <= 0.0 || targetHz <= 0.0) return 0.0;
    return 1200.0 * std::log2(detectedHz / targetHz);
}

double median(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
}

double percentile(std::vector<double> values, double p01)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double idx = std::clamp(p01, 0.0, 1.0) * static_cast<double>(values.size() - 1);
    const size_t i0 = static_cast<size_t>(std::floor(idx));
    const size_t i1 = std::min(values.size() - 1, i0 + 1);
    const double frac = idx - static_cast<double>(i0);
    return values[i0] * (1.0 - frac) + values[i1] * frac;
}

struct CaseResult {
    Waveform waveform = Waveform::Sine;
    LoiaconoRolling::WindowMode windowMode = LoiaconoRolling::WindowMode::RectangularWindow;
    double targetHz = 0.0;
    int evaluationFrames = 0;
    size_t samplesUsed = 0;
    double detectionRate = 0.0;
    double medianAbsRawCents = 0.0;
    double medianAbsSmoothedCents = 0.0;
    double p95AbsSmoothedCents = 0.0;
    double finalSmoothedCents = 0.0;
};

CaseResult runCase(Waveform waveform, LoiaconoRolling::WindowMode windowMode, double targetHz)
{
    LoiaconoRolling transform;
    transform.setComputeMode(LoiaconoRolling::ComputeMode::MultiThread);
    transform.setWindowMode(windowMode);
    transform.setNormalizationMode(LoiaconoRolling::NormalizationMode::Energy);
    transform.setWindowLengthMode(LoiaconoRolling::WindowLengthMode::PeriodMultiple);
    transform.setAlgorithmMode(LoiaconoRolling::AlgorithmMode::Loiacono);
    transform.setLeakiness(0.9995);
    transform.configure(kSampleRate, kFreqMin, kFreqMax, kNumBins, kMultiple);

    const int totalChunks = static_cast<int>(std::ceil((kDurationSeconds * kSampleRate) / kChunkFrames));
    const int warmupChunks = static_cast<int>(std::ceil((kWarmupSeconds * kSampleRate) / kChunkFrames));
    uint64_t sampleCursor = 0;

    std::deque<double> pitchHistory;
    std::vector<double> rawErrors;
    std::vector<double> smoothedErrors;
    std::vector<float> spectrum;
    int evaluationFrames = 0;

    for (int chunk = 0; chunk < totalChunks; ++chunk) {
        const auto audio = makeWaveChunk(waveform, targetHz, 0.85, sampleCursor, kChunkFrames);
        sampleCursor += static_cast<uint64_t>(kChunkFrames);
        transform.processChunk(audio.data(), kChunkFrames);

        if (chunk < warmupChunks) continue;
        evaluationFrames++;

        transform.getSpectrum(spectrum);
        const auto pitch = transform.detectRootPitch(spectrum, kMinPitchHz, kMaxPitchHz);
        if (pitch.confidence <= kConfidenceThreshold || pitch.freqHz <= 0.0) continue;

        const double rawCents = centsError(pitch.freqHz, targetHz);
        rawErrors.push_back(rawCents);

        pitchHistory.push_back(pitch.freqHz);
        if (pitchHistory.size() > kPitchHistorySize) {
            pitchHistory.pop_front();
        }
        std::vector<double> sorted(pitchHistory.begin(), pitchHistory.end());
        const double smoothedHz = median(sorted);
        smoothedErrors.push_back(centsError(smoothedHz, targetHz));
    }

    std::vector<double> absRaw(rawErrors.size(), 0.0);
    std::transform(rawErrors.begin(), rawErrors.end(), absRaw.begin(), [](double v) { return std::abs(v); });
    std::vector<double> absSmoothed(smoothedErrors.size(), 0.0);
    std::transform(smoothedErrors.begin(), smoothedErrors.end(), absSmoothed.begin(), [](double v) { return std::abs(v); });

    CaseResult result;
    result.waveform = waveform;
    result.windowMode = windowMode;
    result.targetHz = targetHz;
    result.evaluationFrames = evaluationFrames;
    result.samplesUsed = smoothedErrors.size();
    result.detectionRate = evaluationFrames > 0
        ? static_cast<double>(result.samplesUsed) / static_cast<double>(evaluationFrames)
        : 0.0;
    result.medianAbsRawCents = median(absRaw);
    result.medianAbsSmoothedCents = median(absSmoothed);
    result.p95AbsSmoothedCents = percentile(absSmoothed, 0.95);
    result.finalSmoothedCents = smoothedErrors.empty() ? 0.0 : smoothedErrors.back();
    return result;
}

const char* windowModeName(LoiaconoRolling::WindowMode mode)
{
    return LoiaconoRolling::windowModeName(mode);
}
} // namespace

int main()
{
    std::cout << "Headless Pitch Tuning Benchmark\n";
    std::cout << "==============================\n";
    std::cout << "sampleRate=" << kSampleRate
              << " bins=" << kNumBins
              << " multiple=" << kMultiple
              << " chunk=" << kChunkFrames
              << " duration=" << kDurationSeconds << "s"
              << " warmup=" << kWarmupSeconds << "s\n\n";

    const std::vector<double> testFreqs = {
        82.4069, 110.0, 146.8324, 196.0, 220.0, 261.6256, 329.6276, 440.0, 659.2551, 880.0
    };
    const std::vector<Waveform> waveforms = {Waveform::Sine, Waveform::Sawtooth, Waveform::Square};
    const std::vector<LoiaconoRolling::WindowMode> windows = {
        LoiaconoRolling::WindowMode::RectangularWindow,
        LoiaconoRolling::WindowMode::HannWindow,
    };

    std::vector<CaseResult> results;
    for (const auto window : windows) {
        for (const auto waveform : waveforms) {
            for (double hz : testFreqs) {
                results.push_back(runCase(waveform, window, hz));
            }
        }
    }

    std::cout << std::left << std::setw(10) << "waveform"
              << std::setw(12) << "window"
              << std::setw(10) << "targetHz"
              << std::setw(10) << "detect%"
              << std::setw(8) << "samples"
              << std::setw(14) << "med|raw|c"
              << std::setw(16) << "med|smooth|c"
              << std::setw(14) << "p95|smooth|c"
              << std::setw(12) << "final|c"
              << "\n";

    bool pass = true;
    double worstMedianSmooth = 0.0;
    std::vector<std::string> warnings;
    for (const auto& r : results) {
        if (r.samplesUsed > 0) {
            worstMedianSmooth = std::max(worstMedianSmooth, r.medianAbsSmoothedCents);
        }
        std::cout << std::left
                  << std::setw(10) << waveformName(r.waveform)
                  << std::setw(12) << windowModeName(r.windowMode)
                  << std::setw(10) << std::fixed << std::setprecision(2) << r.targetHz
                  << std::setw(10) << std::setprecision(1) << (100.0 * r.detectionRate)
                  << std::setw(8) << r.samplesUsed
                  << std::setw(14) << std::setprecision(2) << r.medianAbsRawCents
                  << std::setw(16) << std::setprecision(2) << r.medianAbsSmoothedCents
                  << std::setw(14) << std::setprecision(2) << r.p95AbsSmoothedCents
                  << std::setw(12) << std::showpos << std::setprecision(2) << r.finalSmoothedCents << std::noshowpos
                  << "\n";

        if (r.samplesUsed < 8) {
            warnings.push_back(std::string("low coverage: ") + waveformName(r.waveform)
                               + " " + windowModeName(r.windowMode)
                               + " " + std::to_string(r.targetHz) + "Hz");
            continue;
        }

        // Primary guardrail: this is the user-facing isolated mode path.
        const bool isPrimaryPath = (r.waveform == Waveform::Sawtooth)
            && (r.windowMode == LoiaconoRolling::WindowMode::HannWindow);
        if (isPrimaryPath) {
            if (r.detectionRate < 0.90 || r.medianAbsSmoothedCents > 12.0 || r.p95AbsSmoothedCents > 18.0) {
                pass = false;
            }
        }
    }

    std::cout << "\nWorst median smoothed error: " << std::fixed << std::setprecision(2)
              << worstMedianSmooth << " cents\n";
    if (!warnings.empty()) {
        std::cout << "Warnings (" << warnings.size() << "):\n";
        for (const auto& w : warnings) {
            std::cout << "  - " << w << "\n";
        }
    }
    std::cout << (pass ? "RESULT: PASS\n" : "RESULT: FAIL\n");
    return pass ? 0 : 1;
}
