#pragma once
#include <QWidget>
#include <QImage>
#include <QTimer>
#include <QRect>
#include <QPoint>
#include <QJsonArray>
#include <QString>
#include <vector>
#include "loiacono_rolling.h"

class QPainter;
class QWheelEvent;
class QMouseEvent;

// Combined spectrogram + histogram widget
// Left: scrolling spectrogram (time on X, freq on Y)
// Right: live spectrum histogram (amplitude bars)
// Overlay: runtime stats

class SpectrogramWidget : public QWidget {
    Q_OBJECT
public:
    enum class DisplayNormalizationMode {
        SmoothedGlobalMax,
        PerFrameMax,
        PeakHoldDecay,
        FixedReference,
    };

    enum class ToneCurveMode {
        PowerGamma,
        Smoothstep,
        Sigmoid,
        CustomCurve,
    };

    enum class ColumnFillMode {
        DuplicateSnapshot,
        RollingReconstruction,
    };

    explicit SpectrogramWidget(LoiaconoRolling* transform, QWidget* parent = nullptr);

    // Gradient controls
    void setGain(float g) { gain_ = g; update(); }
    void setGamma(float g) { gamma_ = g; update(); }
    void setFloor(float f) { floor_ = f; update(); }
    float gain() const { return gain_; }
    float gamma() const { return gamma_; }
    float floor() const { return floor_; }
    void setDisplayNormalizationMode(DisplayNormalizationMode mode) { displayNormalizationMode_ = mode; update(); }
    DisplayNormalizationMode displayNormalizationMode() const { return displayNormalizationMode_; }
    void setFixedDisplayReference(float amplitude) { fixedDisplayReference_ = std::max(0.01f, amplitude); update(); }
    float fixedDisplayReference() const { return fixedDisplayReference_; }
    void setToneCurveMode(ToneCurveMode mode) { toneCurveMode_ = mode; update(); }
    ToneCurveMode toneCurveMode() const { return toneCurveMode_; }
    void setCustomToneCurve(const std::vector<QPointF>& controlPoints);
    const std::vector<QPointF>& customToneCurve() const { return customToneCurve_; }
    QJsonArray customToneCurveJson() const;
    void setCustomToneCurveJson(const QJsonArray& curve);
    static const char* displayNormalizationModeName(DisplayNormalizationMode mode);
    static const char* toneCurveModeName(ToneCurveMode mode);
    static const char* columnFillModeName(ColumnFillMode mode);
    void setHardwareAccelerationEnabled(bool enabled);
    bool hardwareAccelerationEnabled() const { return hardwareAccelerationEnabled_; }
    void setColumnFillMode(ColumnFillMode mode) { columnFillMode_ = mode; update(); }
    ColumnFillMode columnFillMode() const { return columnFillMode_; }
    void setRollingReconstructionLimit(int limit) { rollingReconstructionLimit_ = std::max(1, limit); update(); }
    int rollingReconstructionLimit() const { return rollingReconstructionLimit_; }
    void setGridVisible(bool visible) { gridVisible_ = visible; update(); }
    bool gridVisible() const { return gridVisible_; }
    void setBufferEdgeMarkersVisible(bool visible) { bufferEdgeMarkersVisible_ = visible; update(); }
    bool bufferEdgeMarkersVisible() const { return bufferEdgeMarkersVisible_; }
    void setAudioBufferFrames(unsigned int frames) { audioBufferFrames_ = std::max(1u, frames); update(); }
    unsigned int audioBufferFrames() const { return audioBufferFrames_; }
    void setPaused(bool paused);
    bool isPaused() const { return paused_; }
    void setDisplayedTimeSeconds(double seconds);
    double displayedTimeSeconds() const { return displaySeconds_; }
    void resetHistory();
    void onCanvasResized();

    const QImage& spectrogramImage() const { return image_; }
    QImage renderToImage() const;

    // Runtime stats for display
    struct FrameStats {
        double fps = 0;
        double peakHz = 0;
        float peakAmp = 0;
        float maxAmp = 0;
        LoiaconoRolling::PitchResult pitch;
    };
    FrameStats frameStats() const { return frameStats_; }

signals:
    void displayedTimeChanged(double seconds);
    void frequencyRangeChanged(int freqMin, int freqMax);

protected:
    void resizeEvent(QResizeEvent* event) override;
    
private slots:
    void tick();

private:
    class RasterCanvas;
    friend class GlSpectrogramCanvas;
    friend class RasterCanvas;

    struct RGB { uint8_t r, g, b; };
    float visualLevel(float amplitude) const;
    float applyToneCurve(float t) const;
    RGB colormap(float amplitude) const;
    void paintSpectrumColumn(int imageX, const std::vector<float>& spectrum);
    void replaceCanvas();
    void paintContent(QPainter& p, const QSize& canvasSize);
    void paintDecorations(QPainter& p, const QSize& canvasSize);
    QRect spectrogramRect(const QSize& canvasSize) const;
    QRect histogramRect(const QSize& canvasSize) const;
    QRect frequencyAxisRect(const QSize& canvasSize) const;
    QRect timeAxisRect(const QSize& canvasSize) const;
    int binToY(int numBins, const QRect& rect, double binIndex) const;
    void handleWheelZoom(const QPoint& position, int angleDeltaY, const QSize& canvasSize);
    void updateHoverCursor(const QPoint& position, const QSize& canvasSize);
    bool useDirectGpuPipeline() const;

    LoiaconoRolling* transform_;
    QWidget* canvas_ = nullptr;
    QImage image_;        // spectrogram backing store
    QTimer timer_;
    bool hardwareAccelerationEnabled_ = false;
    double displaySeconds_ = 8.0;

    // Gradient parameters
    float gain_ = 1.0f;   // pre-log multiplier
    float gamma_ = 0.6f;  // post-normalize power curve
    float floor_ = 0.05f; // below this = black (noise gate)
    DisplayNormalizationMode displayNormalizationMode_ = DisplayNormalizationMode::SmoothedGlobalMax;
    float fixedDisplayReference_ = 1.0f;
    ToneCurveMode toneCurveMode_ = ToneCurveMode::PowerGamma;
    ColumnFillMode columnFillMode_ = ColumnFillMode::DuplicateSnapshot;
    int rollingReconstructionLimit_ = 24;
    bool gridVisible_ = true;
    bool bufferEdgeMarkersVisible_ = false;
    unsigned int audioBufferFrames_ = 256;
    bool paused_ = false;
    std::vector<QPointF> customToneCurve_;

    float maxAmplitude_ = 1.0f;
    std::vector<float> spectrum_;
    FrameStats frameStats_;

    // FPS tracking
    int frameCount_ = 0;
    qint64 lastFpsTime_ = 0;
    uint64_t lastColumnSampleCount_ = 0;
    double pendingColumnFraction_ = 0.0;
    int pendingGpuColumns_ = 0;
    int historyRevision_ = 0;

    static constexpr int HISTOGRAM_WIDTH = 120; // pixels for the histogram panel
    static constexpr int AXIS_HEIGHT = 18;
    static constexpr int Y_AXIS_WIDTH = 50;   // pixels for frequency labels column
    
    // Pitch detection smoothing
    std::vector<double> pitchHistory_;
    static constexpr size_t MAX_PITCH_HISTORY = 8;
    double smoothedPitchHz_ = 0;
    double displayPitchHz_ = 0.0;
    double displayPitchCents_ = 0.0;
    double displayPitchConfidence_ = 0.0;
    int displayPitchMidiNote_ = -1;
    QString displayPitchNoteName_;
    qint64 lastPitchUpdateMs_ = 0;
    static constexpr qint64 PITCH_PERSIST_MS = 3000;
};
