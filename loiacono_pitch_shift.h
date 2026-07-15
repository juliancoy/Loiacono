#pragma once

#include <cstdint>
#include <vector>
#include <memory>

class LoiaconoRolling;

namespace loiacono {

struct AnalyzeConfig {
    unsigned int sampleRate = 48000;
    double freqMinHz = 100.0;
    double freqMaxHz = 3000.0;
    int binCount = 128;
    int multiple = 24;
    int computeMode = 1;
    int windowMode = 0;
    int normalizationMode = 2;
    int windowLengthMode = 2;
    int algorithmMode = 0;
    double leakiness = 0.9995;
    double baseAFrequencyHz = 440.0;
    bool phaseEnabled = true;
};

class LoiaconoAnalyze {
public:
    explicit LoiaconoAnalyze(const AnalyzeConfig& cfg);
    ~LoiaconoAnalyze();

    void processSample(float sample);
    void getSpectrum(std::vector<float>& out) const;
    void getPhase(std::vector<float>& out) const;

    int numBins() const;
    float latestInputSample() const { return latestInputSample_; }
    uint64_t sampleCount() const { return sampleCount_; }
    const LoiaconoRolling& transform() const;
    LoiaconoRolling& transform();

private:
    std::unique_ptr<LoiaconoRolling> transform_;
    float latestInputSample_ = 0.0f;
    uint64_t sampleCount_ = 0;
};

// Shift complex spectrum energy across Loiacono's log-frequency bins.
// `magnitudes` and `phases` are per-bin polar values from LoiaconoRolling.
void shiftLogSpectrum(const LoiaconoRolling& transform,
                      const std::vector<float>& magnitudes,
                      const std::vector<float>& phases,
                      float ratio,
                      double freqMinHz,
                      double freqMaxHz,
                      std::vector<float>& outMagnitudes,
                      std::vector<float>& outPhases);

// Synthesis mode for Loiacono output from spectral magnitudes/phases.
class LoiaconoSynthesize {
public:
    void reset(int numBins);

    void shiftFromAnalysis(const LoiaconoAnalyze& analyze,
                           const std::vector<float>& magnitudes,
                           const std::vector<float>& phases,
                           float ratio,
                           double freqMinHz,
                           double freqMaxHz);

    float synthSample(const LoiaconoAnalyze& analyze,
                      const std::vector<float>& magnitudes,
                      const std::vector<float>& phases,
                      float amplitudeSmoothing,
                      float phasePull,
                      int controlPeriodSamples = 1);

    float synthShiftedSample(const LoiaconoAnalyze& analyze,
                             float amplitudeSmoothing,
                             float phasePull,
                             int controlPeriodSamples = 1);

    const std::vector<float>& shiftedMagnitudes() const { return shiftedMagnitudes_; }
    const std::vector<float>& shiftedPhases() const { return shiftedPhases_; }
    bool hasShift() const { return hasShift_; }

private:
    std::vector<float> synthAmps_;
    std::vector<double> synthPhase_;
    float synthLevel_ = 0.0f;
    std::vector<float> shiftedMagnitudes_;
    std::vector<float> shiftedPhases_;
    float lastShiftRatio_ = 1.0f;
    bool hasShift_ = false;
};

}  // namespace loiacono
