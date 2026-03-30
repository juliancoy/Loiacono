#pragma once
#include <QWidget>
#include <QImage>
#include <QTimer>
#include <QRect>
#include <QPoint>
#include <vector>
#include "loiacono_rolling.h"

class QPainter;
class QWheelEvent;

// Combined spectrogram + histogram widget
// Left: scrolling spectrogram (time on X, freq on Y)
// Right: live spectrum histogram (amplitude bars)
// Overlay: runtime stats

class SpectrogramWidget : public QWidget {
    Q_OBJECT
public:
    explicit SpectrogramWidget(LoiaconoRolling* transform, QWidget* parent = nullptr);

    // Gradient controls
    void setGain(float g) { gain_ = g; update(); }
    void setGamma(float g) { gamma_ = g; update(); }
    void setFloor(float f) { floor_ = f; update(); }
    float gain() const { return gain_; }
    float gamma() const { return gamma_; }
    float floor() const { return floor_; }
    void setHardwareAccelerationEnabled(bool enabled);
    bool hardwareAccelerationEnabled() const { return hardwareAccelerationEnabled_; }
    void setDisplayedTimeSeconds(double seconds);
    double displayedTimeSeconds() const { return displaySeconds_; }
    void resetHistory();

    const QImage& spectrogramImage() const { return image_; }
    QImage renderToImage() const;

    // Runtime stats for display
    struct FrameStats {
        double fps = 0;
        double peakHz = 0;
        float peakAmp = 0;
        float maxAmp = 0;
    };
    FrameStats frameStats() const { return frameStats_; }

signals:
    void displayedTimeChanged(double seconds);
    void frequencyRangeChanged(int freqMin, int freqMax);

protected:
private slots:
    void tick();

private:
    class RasterCanvas;
    friend class GlSpectrogramCanvas;
    friend class RasterCanvas;

    struct RGB { uint8_t r, g, b; };
    RGB colormap(float amplitude) const;
    void replaceCanvas();
    void paintContent(QPainter& p, const QSize& canvasSize);
    void paintDecorations(QPainter& p, const QSize& canvasSize);
    QRect spectrogramRect(const QSize& canvasSize) const;
    QRect histogramRect(const QSize& canvasSize) const;
    int binToY(int numBins, const QRect& rect, double binIndex) const;
    void handleWheelZoom(const QPoint& position, int angleDeltaY, const QSize& canvasSize);
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

    float maxAmplitude_ = 1.0f;
    std::vector<float> spectrum_;
    FrameStats frameStats_;

    // FPS tracking
    int frameCount_ = 0;
    qint64 lastFpsTime_ = 0;
    qint64 lastColumnTime_ = 0;
    int pendingGpuColumns_ = 0;
    int historyRevision_ = 0;

    static constexpr int HISTOGRAM_WIDTH = 120; // pixels for the histogram panel
    static constexpr int AXIS_HEIGHT = 18;
};
