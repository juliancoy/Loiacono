#pragma once

#include <memory>
#include <vector>

class LoiaconoRolling;

namespace jtune {

struct PitchTrackerOptions {
    unsigned int sampleRate = 48000;
    int minMidi = 40;
    int maxMidi = 84;
    int multiple = 24;
    int binCount = 160;
    int analysisHop = 128;
    float voicedThreshold = 0.20f;

    double freqMinHz = 100.0;
    double freqMaxHz = 3000.0;
    double leakiness = 0.9995;
    double baseAFrequencyHz = 440.0;

    int computeMode = 1;
    int windowMode = 0;
    int normalizationMode = 2;
    int windowLengthMode = 2;
    int algorithmMode = 0;
};

class RollingPitchTracker {
public:
    explicit RollingPitchTracker(const PitchTrackerOptions& opts);
    ~RollingPitchTracker();

    bool processSample(float sample);

    double pitchHz() const { return pitchHz_; }
    float confidence() const { return confidence_; }

private:
    PitchTrackerOptions opts_;
    std::unique_ptr<LoiaconoRolling> transform_;
    std::vector<float> spectrum_;
    int analysisCountdown_ = 0;

    double pitchHz_ = 0.0;
    float confidence_ = 0.0f;
};

}  // namespace jtune
