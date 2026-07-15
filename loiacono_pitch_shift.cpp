#include "loiacono_pitch_shift.h"

#include "loiacono_rolling.h"

#include <algorithm>
#include <cmath>

namespace loiacono {
namespace {

constexpr double kTwoPi = 2.0 * M_PI;

}  // namespace

LoiaconoAnalyze::LoiaconoAnalyze(const AnalyzeConfig& cfg)
    : transform_(std::make_unique<LoiaconoRolling>())
{
    transform_->setComputeMode(static_cast<LoiaconoRolling::ComputeMode>(cfg.computeMode));
    transform_->setWindowMode(static_cast<LoiaconoRolling::WindowMode>(cfg.windowMode));
    transform_->setNormalizationMode(static_cast<LoiaconoRolling::NormalizationMode>(cfg.normalizationMode));
    transform_->setWindowLengthMode(static_cast<LoiaconoRolling::WindowLengthMode>(cfg.windowLengthMode));
    transform_->setAlgorithmMode(static_cast<LoiaconoRolling::AlgorithmMode>(cfg.algorithmMode));
    transform_->setLeakiness(cfg.leakiness);
    transform_->setBaseAFrequency(cfg.baseAFrequencyHz);
    transform_->setPhaseCalculationEnabled(cfg.phaseEnabled);
    transform_->configure(cfg.sampleRate, cfg.freqMinHz, cfg.freqMaxHz, cfg.binCount, cfg.multiple);
}

LoiaconoAnalyze::~LoiaconoAnalyze() = default;

void LoiaconoAnalyze::processSample(float sample)
{
    latestInputSample_ = sample;
    transform_->processSample(sample);
    ++sampleCount_;
}

void LoiaconoAnalyze::getSpectrum(std::vector<float>& out) const
{
    transform_->getSpectrum(out);
}

void LoiaconoAnalyze::getPhase(std::vector<float>& out) const
{
    transform_->getPhase(out);
}

int LoiaconoAnalyze::numBins() const
{
    return transform_->numBins();
}

const LoiaconoRolling& LoiaconoAnalyze::transform() const
{
    return *transform_;
}

LoiaconoRolling& LoiaconoAnalyze::transform()
{
    return *transform_;
}

void shiftLogSpectrum(const LoiaconoRolling& transform,
                      const std::vector<float>& magnitudes,
                      const std::vector<float>& phases,
                      float ratio,
                      double freqMinHz,
                      double freqMaxHz,
                      std::vector<float>& outMagnitudes,
                      std::vector<float>& outPhases)
{
    const int numBins = transform.numBins();
    const size_t n = static_cast<size_t>(std::max(0, numBins));
    outMagnitudes.assign(n, 0.0f);
    outPhases.assign(n, 0.0f);
    if (numBins <= 0) return;

    std::vector<float> outReal(n, 0.0f);
    std::vector<float> outImag(n, 0.0f);

    // A constant-Q analysis of one sinusoid lights up several neighboring
    // bins. Resynthesizing every one of those bins as an independent
    // oscillator turns the analysis kernel's sidelobes into audible tones.
    // Keep spectral peaks (partials) and reject the leakage skirt/noise floor.
    const float maxMagnitude = magnitudes.empty()
        ? 0.0f
        : *std::max_element(magnitudes.begin(), magnitudes.end());
    const float peakFloor = maxMagnitude * 0.01f;

    for (int i = 0; i < numBins; ++i) {
        const float mag = (i < static_cast<int>(magnitudes.size())) ? magnitudes[static_cast<size_t>(i)] : 0.0f;
        const float left = i > 0 && i - 1 < static_cast<int>(magnitudes.size())
            ? magnitudes[static_cast<size_t>(i - 1)]
            : 0.0f;
        const float right = i + 1 < static_cast<int>(magnitudes.size())
            ? magnitudes[static_cast<size_t>(i + 1)]
            : 0.0f;
        if (mag < peakFloor || mag < left || mag < right) continue;

        const double hz = transform.binFreqHz(i);
        const double shiftedHz = hz * static_cast<double>(ratio);
        if (shiftedHz < freqMinHz || shiftedHz > freqMaxHz) continue;

        const double dstBin = transform.freqToBin(shiftedHz);
        const int i0 = std::clamp(static_cast<int>(std::floor(dstBin)), 0, numBins - 1);
        const int i1 = std::min(numBins - 1, i0 + 1);
        const float frac = static_cast<float>(dstBin - static_cast<double>(i0));

        const float ph = (i < static_cast<int>(phases.size())) ? phases[static_cast<size_t>(i)] : 0.0f;
        const float re = mag * std::cos(ph);
        const float im = mag * std::sin(ph);

        outReal[static_cast<size_t>(i0)] += re * (1.0f - frac);
        outImag[static_cast<size_t>(i0)] += im * (1.0f - frac);
        outReal[static_cast<size_t>(i1)] += re * frac;
        outImag[static_cast<size_t>(i1)] += im * frac;
    }

    for (int i = 0; i < numBins; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const float re = outReal[idx];
        const float im = outImag[idx];
        outMagnitudes[idx] = std::sqrt(re * re + im * im);
        outPhases[idx] = std::atan2(im, re);
    }
}

void LoiaconoSynthesize::reset(int numBins)
{
    const size_t n = static_cast<size_t>(std::max(0, numBins));
    synthAmps_.assign(n, 0.0f);
    synthPhase_.assign(n, 0.0);
    synthLevel_ = 0.0f;
    shiftedMagnitudes_.assign(n, 0.0f);
    shiftedPhases_.assign(n, 0.0f);
    lastShiftRatio_ = 1.0f;
    hasShift_ = false;
}

void LoiaconoSynthesize::shiftFromAnalysis(const LoiaconoAnalyze& analyze,
                                           const std::vector<float>& magnitudes,
                                           const std::vector<float>& phases,
                                           float ratio,
                                           double freqMinHz,
                                           double freqMaxHz)
{
    lastShiftRatio_ = ratio;
    hasShift_ = true;
    shiftLogSpectrum(analyze.transform(), magnitudes, phases, ratio, freqMinHz, freqMaxHz, shiftedMagnitudes_, shiftedPhases_);
}

float LoiaconoSynthesize::synthSample(const LoiaconoAnalyze& analyze,
                                      const std::vector<float>& magnitudes,
                                      const std::vector<float>& phases,
                                      float amplitudeSmoothing,
                                      float phasePull,
                                      int controlPeriodSamples)
{
    const LoiaconoRolling& transform = analyze.transform();
    const int numBins = transform.numBins();
    if (numBins <= 0) return 0.0f;

    if (synthAmps_.size() != static_cast<size_t>(numBins) || synthPhase_.size() != static_cast<size_t>(numBins)) {
        reset(numBins);
    }

    const float ampFrameMix = std::clamp(amplitudeSmoothing, 0.01f, 1.0f);
    const float phaseFrameMix = std::clamp(phasePull, 0.0f, 1.0f);
    const float period = static_cast<float>(std::max(1, controlPeriodSamples));
    const float ampMix = 1.0f - std::pow(1.0f - ampFrameMix, 1.0f / period);
    const float phaseMix = 1.0f - std::pow(1.0f - phaseFrameMix, 1.0f / period);

    float totalMag = 0.0f;
    for (int i = 0; i < numBins; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const float mag = (i < static_cast<int>(magnitudes.size())) ? magnitudes[idx] : 0.0f;
        const float ph = (i < static_cast<int>(phases.size())) ? phases[idx] : 0.0f;

        synthAmps_[idx] = synthAmps_[idx] * (1.0f - ampMix) + mag * ampMix;
        totalMag += synthAmps_[idx];

        if (synthAmps_[idx] > 1e-7f) {
            // Transform phases are coefficients in an absolute-time Fourier
            // basis, not the phase of an oscillator at this sample. Advance
            // the coefficient phase to "now" before phase-locking the
            // oscillator; otherwise every analysis update pulls it backward
            // by omega * sampleCount and creates hop-rate phase modulation.
            const double omega = kTwoPi * transform.binFreqHz(i) / transform.sampleRate();
            const double sampleIndex = analyze.sampleCount() > 0
                ? static_cast<double>(analyze.sampleCount() - 1)
                : 0.0;
            const double targetPhase = static_cast<double>(ph) + omega * sampleIndex;
            const double diff = std::remainder(targetPhase - synthPhase_[idx], kTwoPi);
            synthPhase_[idx] += static_cast<double>(phaseMix) * diff;
        }
    }

    const float levelTarget = std::clamp(totalMag / static_cast<float>(numBins), 0.01f, 20.0f);
    synthLevel_ = synthLevel_ * 0.95f + levelTarget * 0.05f;

    double accum = 0.0;
    for (int i = 0; i < numBins; ++i) {
        const size_t idx = static_cast<size_t>(i);
        const float amp = synthAmps_[idx];
        if (amp < 1e-7f) continue;
        accum += static_cast<double>(amp) * std::cos(synthPhase_[idx]);
        const double omega = kTwoPi * transform.binFreqHz(i) / transform.sampleRate();
        synthPhase_[idx] += omega;
        if (synthPhase_[idx] > kTwoPi) synthPhase_[idx] -= kTwoPi;
    }

    float out = static_cast<float>(accum / static_cast<double>(numBins));
    out /= std::max(0.02f, synthLevel_);
    return std::clamp(out, -1.0f, 1.0f);
}

float LoiaconoSynthesize::synthShiftedSample(const LoiaconoAnalyze& analyze,
                                             float amplitudeSmoothing,
                                             float phasePull,
                                             int controlPeriodSamples)
{
    if (hasShift_ && std::abs(lastShiftRatio_ - 1.0f) < 1e-6f) {
        return analyze.latestInputSample();
    }
    return synthSample(
        analyze, shiftedMagnitudes_, shiftedPhases_, amplitudeSmoothing, phasePull, controlPeriodSamples);
}

}  // namespace loiacono
