#include "pitch_tracker.h"

#include "loiacono_rolling.h"

#include <algorithm>
#include <cmath>

namespace jtune {
namespace {

float midiToHz(double midi)
{
    return static_cast<float>(440.0 * std::pow(2.0, (midi - 69.0) / 12.0));
}

}  // namespace

RollingPitchTracker::RollingPitchTracker(const PitchTrackerOptions& opts)
    : opts_(opts), transform_(std::make_unique<LoiaconoRolling>())
{
    opts_.binCount = std::clamp(opts_.binCount, 32, 2400);
    opts_.analysisHop = std::max(16, opts_.analysisHop);
    opts_.multiple = std::clamp(opts_.multiple, 2, 240);
    opts_.leakiness = std::clamp(opts_.leakiness, 0.99, 1.0);
    opts_.baseAFrequencyHz = std::clamp(opts_.baseAFrequencyHz, 400.0, 500.0);

    if (opts_.freqMinHz <= 0.0) opts_.freqMinHz = midiToHz(opts_.minMidi);
    if (opts_.freqMaxHz <= opts_.freqMinHz) opts_.freqMaxHz = midiToHz(opts_.maxMidi);
    opts_.freqMinHz = std::clamp(opts_.freqMinHz, 20.0, 12000.0);
    opts_.freqMaxHz = std::clamp(opts_.freqMaxHz, 40.0, 20000.0);
    if (opts_.freqMinHz >= opts_.freqMaxHz - 10.0) {
        opts_.freqMaxHz = opts_.freqMinHz + 10.0;
    }

    transform_->setComputeMode(static_cast<LoiaconoRolling::ComputeMode>(opts_.computeMode));
    transform_->setWindowMode(static_cast<LoiaconoRolling::WindowMode>(opts_.windowMode));
    transform_->setNormalizationMode(static_cast<LoiaconoRolling::NormalizationMode>(opts_.normalizationMode));
    transform_->setWindowLengthMode(static_cast<LoiaconoRolling::WindowLengthMode>(opts_.windowLengthMode));
    transform_->setAlgorithmMode(static_cast<LoiaconoRolling::AlgorithmMode>(opts_.algorithmMode));
    transform_->setLeakiness(opts_.leakiness);
    transform_->setBaseAFrequency(opts_.baseAFrequencyHz);
    transform_->configure(opts_.sampleRate, opts_.freqMinHz, opts_.freqMaxHz, opts_.binCount, opts_.multiple);

    spectrum_.assign(static_cast<size_t>(std::max(1, transform_->numBins())), 0.0f);
    analysisCountdown_ = opts_.analysisHop;
}

RollingPitchTracker::~RollingPitchTracker() = default;

bool RollingPitchTracker::processSample(float sample)
{
    transform_->processSample(sample);

    if (--analysisCountdown_ > 0) {
        return false;
    }
    analysisCountdown_ = opts_.analysisHop;

    transform_->getSpectrum(spectrum_);
    const auto pitch = transform_->detectRootPitch(
        spectrum_,
        opts_.freqMinHz,
        opts_.freqMaxHz,
        opts_.baseAFrequencyHz);

    confidence_ = static_cast<float>(pitch.confidence);
    if (pitch.freqHz > 0.0 && confidence_ >= opts_.voicedThreshold) {
        pitchHz_ = pitch.freqHz;
    } else {
        pitchHz_ = 0.0;
    }

    return true;
}

}  // namespace jtune
