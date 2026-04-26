#include "spectrogram_widget.h"
#include "gl_spectrogram_canvas.h"

#include <QDateTime>
#include <QHBoxLayout>
#include <QPainter>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <QMouseEvent>
#include <QJsonObject>
#include <algorithm>
#include <cmath>
#include <cstring>

class SpectrogramWidget::RasterCanvas : public QWidget {
public:
    explicit RasterCanvas(SpectrogramWidget* owner) : QWidget(owner), owner_(owner)
    {
        setMouseTracking(true);
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        owner_->paintContent(p, size());
    }

    void resizeEvent(QResizeEvent*) override
    {
        owner_->onCanvasResized();
    }

    void wheelEvent(QWheelEvent* event) override
    {
        owner_->handleWheelZoom(event->position().toPoint(), event->angleDelta().y(), size());
        event->accept();
    }

    void mouseMoveEvent(QMouseEvent* event) override
    {
        owner_->updateHoverCursor(event->position().toPoint(), size());
        QWidget::mouseMoveEvent(event);
    }

    void leaveEvent(QEvent* event) override
    {
        unsetCursor();
        QWidget::leaveEvent(event);
    }

private:
    SpectrogramWidget* owner_;
};

SpectrogramWidget::SpectrogramWidget(LoiaconoRolling* transform, QWidget* parent)
    : QWidget(parent), transform_(transform)
{
    setMinimumSize(600, 300);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);  // Fill available space

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    replaceCanvas();

    image_ = QImage(1, 1, QImage::Format_RGB32);
    image_.fill(Qt::black);
    customToneCurve_ = {
        QPointF(0.0, 0.0),
        QPointF(0.25, 0.18),
        QPointF(0.5, 0.5),
        QPointF(0.75, 0.82),
        QPointF(1.0, 1.0),
    };
    lastFpsTime_ = QDateTime::currentMSecsSinceEpoch();

    connect(&timer_, &QTimer::timeout, this, &SpectrogramWidget::tick);
    timer_.start(0);
}

void SpectrogramWidget::setPaused(bool paused)
{
    if (paused_ == paused) return;
    paused_ = paused;
    lastColumnSampleCount_ = transform_->getStats().totalSamples;
    pendingColumnFraction_ = 0.0;
    pendingGpuColumns_ = 0;
    update();
}

const char* SpectrogramWidget::displayNormalizationModeName(DisplayNormalizationMode mode)
{
    switch (mode) {
    case DisplayNormalizationMode::SmoothedGlobalMax:
        return "smoothed";
    case DisplayNormalizationMode::PerFrameMax:
        return "per-frame";
    case DisplayNormalizationMode::PeakHoldDecay:
        return "peak-hold";
    case DisplayNormalizationMode::FixedReference:
        return "fixed";
    }
    return "unknown";
}

const char* SpectrogramWidget::toneCurveModeName(ToneCurveMode mode)
{
    switch (mode) {
    case ToneCurveMode::PowerGamma:
        return "power";
    case ToneCurveMode::Smoothstep:
        return "smoothstep";
    case ToneCurveMode::Sigmoid:
        return "sigmoid";
    case ToneCurveMode::CustomCurve:
        return "custom";
    }
    return "unknown";
}

const char* SpectrogramWidget::columnFillModeName(ColumnFillMode mode)
{
    switch (mode) {
    case ColumnFillMode::DuplicateSnapshot:
        return "duplicate";
    case ColumnFillMode::RollingReconstruction:
        return "rolling";
    }
    return "unknown";
}

void SpectrogramWidget::setCustomToneCurve(const std::vector<QPointF>& controlPoints)
{
    if (controlPoints.size() < 2) return;

    customToneCurve_ = controlPoints;
    std::sort(customToneCurve_.begin(), customToneCurve_.end(), [](const QPointF& a, const QPointF& b) {
        return a.x() < b.x();
    });
    customToneCurve_.front().setX(0.0);
    customToneCurve_.front().setY(std::clamp(customToneCurve_.front().y(), 0.0, 1.0));
    customToneCurve_.back().setX(1.0);
    customToneCurve_.back().setY(std::clamp(customToneCurve_.back().y(), 0.0, 1.0));
    for (auto& point : customToneCurve_) {
        point.setX(std::clamp(point.x(), 0.0, 1.0));
        point.setY(std::clamp(point.y(), 0.0, 1.0));
    }
    update();
}

QJsonArray SpectrogramWidget::customToneCurveJson() const
{
    QJsonArray arr;
    for (const auto& point : customToneCurve_) {
        arr.append(QJsonObject{
            {"x", point.x()},
            {"y", point.y()},
        });
    }
    return arr;
}

void SpectrogramWidget::setCustomToneCurveJson(const QJsonArray& curve)
{
    std::vector<QPointF> points;
    points.reserve(curve.size());
    for (const auto& value : curve) {
        if (!value.isObject()) continue;
        QJsonObject obj = value.toObject();
        points.emplace_back(obj.value("x").toDouble(), obj.value("y").toDouble());
    }
    if (points.size() >= 2) {
        setCustomToneCurve(points);
    }
}

void SpectrogramWidget::setHardwareAccelerationEnabled(bool enabled)
{
    if (hardwareAccelerationEnabled_ == enabled) return;
    hardwareAccelerationEnabled_ = enabled;
    replaceCanvas();
}

void SpectrogramWidget::setDisplayedTimeSeconds(double seconds)
{
    double clampedSeconds = std::max(0.5, seconds);
    if (std::abs(displaySeconds_ - clampedSeconds) < 1e-9) return;
    displaySeconds_ = clampedSeconds;
    lastColumnSampleCount_ = transform_->getStats().totalSamples;
    pendingColumnFraction_ = 0.0;
    pendingGpuColumns_ = 0;
    update();
    emit displayedTimeChanged(displaySeconds_);
}

void SpectrogramWidget::resetHistory()
{
    image_.fill(Qt::black);
    maxAmplitude_ = 1.0f;
    frameStats_.peakAmp = 0.0f;
    frameStats_.maxAmp = maxAmplitude_;
    pitchHistory_.clear();
    smoothedPitchHz_ = 0.0;
    frameStats_.pitch = {};
    displayPitchHz_ = 0.0;
    displayPitchCents_ = 0.0;
    displayPitchConfidence_ = 0.0;
    displayPitchMidiNote_ = -1;
    displayPitchNoteName_.clear();
    lastPitchUpdateMs_ = 0;
    lastColumnSampleCount_ = transform_->getStats().totalSamples;
    pendingColumnFraction_ = 0.0;
    pendingGpuColumns_ = 0;
    historyRevision_++;
    if (canvas_) canvas_->update();
}

void SpectrogramWidget::onCanvasResized()
{
    if (!canvas_) return;
    
    QRect spectRect = spectrogramRect(canvas_->size());
    
    
    // Resize the image to match new canvas size
    if (image_.width() != spectRect.width() || image_.height() != spectRect.height()) {
        // Start fresh with a black image - don't copy old data because the
        // vertical scaling may have changed (bins need to stretch to new height)
        image_ = QImage(spectRect.width(), spectRect.height(), QImage::Format_RGB32);
        image_.fill(Qt::black);
        maxAmplitude_ = 1.0f;
        frameStats_.peakAmp = 0.0f;
        frameStats_.maxAmp = maxAmplitude_;
    }
    
    // Trigger repaint after image is resized
    canvas_->update();
    
    lastColumnSampleCount_ = transform_->getStats().totalSamples;
    pendingColumnFraction_ = 0.0;
    pendingGpuColumns_ = 0;
    historyRevision_++;
    
    canvas_->update();
}

void SpectrogramWidget::resizeEvent(QResizeEvent* event)
{
    QWidget::resizeEvent(event);
    
    // Resize the image to match new size and trigger repaint
    if (canvas_) {
        onCanvasResized();
        // onCanvasResized now calls canvas_->update() after resizing the image
    }
}

void SpectrogramWidget::replaceCanvas()
{
    auto* layout = qobject_cast<QHBoxLayout*>(this->layout());
    if (!layout) return;

    if (canvas_) {
        layout->removeWidget(canvas_);
        canvas_->deleteLater();
        canvas_ = nullptr;
    }

    if (hardwareAccelerationEnabled_) {
        auto* glCanvas = new GlSpectrogramCanvas(this);
        glCanvas->setAutoFillBackground(false);
        canvas_ = glCanvas;
    } else {
        canvas_ = new RasterCanvas(this);
    }

    layout->addWidget(canvas_, 1);  // Add with stretch factor 1 to fill space
    canvas_->setMinimumSize(100, 100);  // Lower minimum to allow flexibility
    canvas_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

QRect SpectrogramWidget::spectrogramRect(const QSize& canvasSize) const
{
    // Reserve space on left for Y-axis (frequency labels) and on right for histogram
    int w = std::max(1, canvasSize.width() - HISTOGRAM_WIDTH - Y_AXIS_WIDTH);
    int h = std::max(1, canvasSize.height() - AXIS_HEIGHT);
    return QRect(Y_AXIS_WIDTH, 0, w, h);
}

QRect SpectrogramWidget::histogramRect(const QSize& canvasSize) const
{
    QRect spectRect = spectrogramRect(canvasSize);
    return QRect(spectRect.right() + 2, spectRect.top(), HISTOGRAM_WIDTH - 2, spectRect.height());
}

QRect SpectrogramWidget::frequencyAxisRect(const QSize& canvasSize) const
{
    QRect spectRect = spectrogramRect(canvasSize);
    return QRect(0, spectRect.top(), Y_AXIS_WIDTH - 5, spectRect.height());
}

QRect SpectrogramWidget::timeAxisRect(const QSize& canvasSize) const
{
    QRect spectRect = spectrogramRect(canvasSize);
    return QRect(spectRect.left(), spectRect.bottom() + 1, spectRect.width(), AXIS_HEIGHT);
}

int SpectrogramWidget::binToY(int numBins, const QRect& rect, double binIndex) const
{
    if (numBins <= 1) return rect.bottom();
    double t = std::clamp(binIndex / (numBins - 1.0), 0.0, 1.0);
    return rect.bottom() - static_cast<int>(std::lround(t * (rect.height() - 1)));
}

bool SpectrogramWidget::useDirectGpuPipeline() const
{
    if (!hardwareAccelerationEnabled_) return false;
    if (transform_->algorithmMode() != LoiaconoRolling::AlgorithmMode::Loiacono) return false;
    if (columnFillMode_ != ColumnFillMode::DuplicateSnapshot) return false;
    if (transform_->activeComputeMode() != LoiaconoRolling::ComputeMode::GpuCompute) return false;
    if (displayNormalizationMode_ != DisplayNormalizationMode::SmoothedGlobalMax) return false;
    if (toneCurveMode_ != ToneCurveMode::PowerGamma) return false;
    auto windowMode = transform_->windowMode();
    return windowMode == LoiaconoRolling::WindowMode::RectangularWindow
        || windowMode == LoiaconoRolling::WindowMode::LeakyWindow;
}

void SpectrogramWidget::paintSpectrumColumn(int imageX, const std::vector<float>& spectrum)
{
    int nb = transform_->numBins();
    int h = image_.height();
    const int spectrumBins = static_cast<int>(spectrum.size());
    const int usableBins = std::min(nb, spectrumBins);
    if (imageX < 0 || imageX >= image_.width() || h <= 0 || usableBins <= 0) return;

    for (int row = 0; row < h; ++row) {
        double binF = (usableBins - 1) * (h - 1 - row) / std::max(1, h - 1);
        int fi = static_cast<int>(std::clamp(std::floor(binF), 0.0, static_cast<double>(usableBins - 1)));
        auto [r, g, b] = colormap(spectrum[fi]);
        auto* line = reinterpret_cast<QRgb*>(image_.scanLine(row));
        line[imageX] = qRgb(r, g, b);
    }
}

void SpectrogramWidget::handleWheelZoom(const QPoint& position, int angleDeltaY, const QSize& canvasSize)
{
    if (angleDeltaY == 0) return;

    QRect spectRect = spectrogramRect(canvasSize);
    QRect xAxisRect = timeAxisRect(canvasSize);
    QRect yAxisHotRect = frequencyAxisRect(canvasSize);

    double zoomFactor = angleDeltaY > 0 ? 0.85 : 1.18;

    if (xAxisRect.contains(position)) {
        double oldSpan = displaySeconds_;
        double newSpan = std::clamp(oldSpan * zoomFactor, 0.5, 30.0);
        double anchor = spectRect.width() > 1
            ? std::clamp((position.x() - spectRect.left()) / static_cast<double>(spectRect.width() - 1), 0.0, 1.0)
            : 1.0;
        double leftAge = (1.0 - anchor) * oldSpan;
        double rightAge = anchor * oldSpan;
        double scaledLeft = leftAge * (newSpan / oldSpan);
        double scaledRight = rightAge * (newSpan / oldSpan);
        double anchoredSpan = scaledLeft + scaledRight;
        setDisplayedTimeSeconds(anchoredSpan);
        emit displayedTimeChanged(displaySeconds_);
        return;
    }

    if (yAxisHotRect.contains(position)) {
        auto stats = transform_->getStats();
        double curMin = stats.freqMin;
        double curMax = stats.freqMax;
        if (curMin <= 0 || curMax <= curMin) return;

        double t = spectRect.height() > 1
            ? std::clamp((spectRect.bottom() - position.y()) / static_cast<double>(spectRect.height() - 1), 0.0, 1.0)
            : 0.5;
        double logMin = std::log(curMin);
        double logMax = std::log(curMax);
        double anchorLogHz = logMin + t * (logMax - logMin);
        double newLogSpan = (logMax - logMin) * zoomFactor;

        double minSpan = std::log(1.08);
        double maxSpan = std::log(24000.0 / 20.0);
        newLogSpan = std::clamp(newLogSpan, minSpan, maxSpan);

        double newLogMin = anchorLogHz - t * newLogSpan;
        double newLogMax = newLogMin + newLogSpan;

        double minAllowed = std::log(20.0);
        double maxAllowed = std::log(transform_->sampleRate() * 0.49);
        if (newLogMin < minAllowed) {
            newLogMin = minAllowed;
            newLogMax = newLogMin + newLogSpan;
        }
        if (newLogMax > maxAllowed) {
            newLogMax = maxAllowed;
            newLogMin = newLogMax - newLogSpan;
        }

        int newMin = static_cast<int>(std::lround(std::exp(newLogMin)));
        int newMax = static_cast<int>(std::lround(std::exp(newLogMax)));
        newMin = std::clamp(newMin, 20, static_cast<int>(transform_->sampleRate() * 0.49) - 50);
        newMax = std::clamp(newMax, newMin + 50, static_cast<int>(transform_->sampleRate() * 0.49));

        emit frequencyRangeChanged(newMin, newMax);
        return;
    }

}

void SpectrogramWidget::updateHoverCursor(const QPoint& position, const QSize& canvasSize)
{
    if (!canvas_) return;
    if (frequencyAxisRect(canvasSize).contains(position)) {
        canvas_->setCursor(Qt::SizeVerCursor);
    } else if (timeAxisRect(canvasSize).contains(position)) {
        canvas_->setCursor(Qt::SizeHorCursor);
    } else {
        canvas_->unsetCursor();
    }
}

void SpectrogramWidget::tick()
{
    if (paused_) return;

    int nb = transform_->numBins();
    if (nb < 1 || !canvas_) return;

    QRect spectRect = spectrogramRect(canvas_->size());

    if (image_.width() != spectRect.width() || image_.height() != spectRect.height()) {
        // Resize happened - start fresh with correct vertical scaling
        image_ = QImage(spectRect.width(), spectRect.height(), QImage::Format_RGB32);
        image_.fill(Qt::black);
        lastColumnSampleCount_ = transform_->getStats().totalSamples;
        pendingColumnFraction_ = 0.0;
        pendingGpuColumns_ = 0;
        historyRevision_++;
    }

    if (!useDirectGpuPipeline()) {
        transform_->getSpectrum(spectrum_);
    } else {
        spectrum_.clear();
    }

    float currentMax = frameStats_.peakAmp;
    int peakIdx = 0;
    // In GPU modes, peak is computed on GPU. In CPU modes, compute it here.
    if (!useDirectGpuPipeline()) {
        currentMax = 0;
        for (int i = 0; i < static_cast<int>(spectrum_.size()); i++) {
            if (spectrum_[i] > currentMax) {
                currentMax = spectrum_[i];
                peakIdx = i;
            }
        }
    }
    if (!useDirectGpuPipeline()) {
        switch (displayNormalizationMode_) {
        case DisplayNormalizationMode::SmoothedGlobalMax:
            if (currentMax > maxAmplitude_) {
                maxAmplitude_ = currentMax;
            } else {
                maxAmplitude_ = maxAmplitude_ * 0.997f + currentMax * 0.003f;
            }
            break;
        case DisplayNormalizationMode::PerFrameMax:
            maxAmplitude_ = currentMax;
            break;
        case DisplayNormalizationMode::PeakHoldDecay:
            if (currentMax > maxAmplitude_) {
                maxAmplitude_ = currentMax;
            } else {
                maxAmplitude_ *= 0.995f;
            }
            break;
        case DisplayNormalizationMode::FixedReference:
            maxAmplitude_ = fixedDisplayReference_;
            break;
        }
        if (maxAmplitude_ < 0.01f) maxAmplitude_ = 0.01f;
    }

    int w = image_.width();
    int h = image_.height();
    auto runtimeStats = transform_->getStats();
    if (lastColumnSampleCount_ == 0) {
        lastColumnSampleCount_ = runtimeStats.totalSamples;
    }
    const uint64_t previousColumnSampleCount = lastColumnSampleCount_;
    uint64_t newSamples = runtimeStats.totalSamples >= lastColumnSampleCount_
        ? (runtimeStats.totalSamples - lastColumnSampleCount_)
        : 0;
    lastColumnSampleCount_ = runtimeStats.totalSamples;
    double samplesPerColumn = std::max(1.0, (displaySeconds_ * transform_->sampleRate()) / std::max(1, w));
    pendingColumnFraction_ += static_cast<double>(newSamples) / samplesPerColumn;
    int columnsToAdvance = static_cast<int>(std::floor(pendingColumnFraction_));
    columnsToAdvance = std::clamp(columnsToAdvance, 0, std::max(1, w));
    if (columnsToAdvance > 0) {
        pendingColumnFraction_ -= columnsToAdvance;
    }
    pendingGpuColumns_ = useDirectGpuPipeline() ? columnsToAdvance : 0;

    if (columnsToAdvance > 0 && !useDirectGpuPipeline()) {
        for (int y = 0; y < h; y++) {
            auto* line = reinterpret_cast<QRgb*>(image_.scanLine(y));
            if (columnsToAdvance < w) {
                std::memmove(line, line + columnsToAdvance, (w - columnsToAdvance) * sizeof(QRgb));
            }
        }
    }

    if (columnsToAdvance > 0 && !useDirectGpuPipeline()) {
        if (columnFillMode_ == ColumnFillMode::RollingReconstruction
            && newSamples > 0
            && previousColumnSampleCount > 0) {
            const int reconstructedColumns = std::min(columnsToAdvance, rollingReconstructionLimit_);
            std::vector<uint64_t> sampleCounts;
            sampleCounts.reserve(static_cast<size_t>(reconstructedColumns));
            for (int i = 0; i < reconstructedColumns; ++i) {
                const double t = static_cast<double>(i + 1) / static_cast<double>(reconstructedColumns);
                const uint64_t offset = static_cast<uint64_t>(std::llround(t * static_cast<double>(newSamples)));
                sampleCounts.push_back(std::min(runtimeStats.totalSamples, previousColumnSampleCount + offset));
            }
            std::vector<std::vector<float>> spectra;
            transform_->getSpectraAtSampleCounts(sampleCounts, spectra);
            if (spectra.size() == sampleCounts.size()) {
                for (int i = 0; i < columnsToAdvance; ++i) {
                    int spectrumIx = 0;
                    if (columnsToAdvance > 1 && reconstructedColumns > 1) {
                        const double t = static_cast<double>(i) / static_cast<double>(columnsToAdvance - 1);
                        spectrumIx = static_cast<int>(std::lround(t * static_cast<double>(reconstructedColumns - 1)));
                    }
                    spectrumIx = std::clamp(spectrumIx, 0, reconstructedColumns - 1);
                    paintSpectrumColumn(w - columnsToAdvance + i, spectra[static_cast<size_t>(spectrumIx)]);
                }
            } else {
                for (int x = w - columnsToAdvance; x < w; ++x) {
                    paintSpectrumColumn(x, spectrum_);
                }
            }
        } else {
            for (int x = w - columnsToAdvance; x < w; ++x) {
                paintSpectrumColumn(x, spectrum_);
            }
        }
    }

    frameCount_++;
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - lastFpsTime_ >= 500) {
        frameStats_.fps = frameCount_ * 1000.0 / (now - lastFpsTime_);
        frameCount_ = 0;
        lastFpsTime_ = now;
    }
    if (!useDirectGpuPipeline()) {
        frameStats_.peakHz = nb > 0 ? transform_->binFreqHz(peakIdx) : 0;
        frameStats_.peakAmp = currentMax;
        frameStats_.maxAmp = maxAmplitude_;
        
        // Detect root pitch using harmonic correlation
        auto pitch = transform_->detectRootPitch(spectrum_, 50.0, 2000.0);
        
        // Apply smoothing for stable display
        if (pitch.confidence > 0.3) {
            pitchHistory_.push_back(pitch.freqHz);
            if (pitchHistory_.size() > MAX_PITCH_HISTORY) {
                pitchHistory_.erase(pitchHistory_.begin());
            }
            // Use median for robust smoothing
            std::vector<double> sorted = pitchHistory_;
            std::sort(sorted.begin(), sorted.end());
            smoothedPitchHz_ = sorted[sorted.size() / 2];
            
            // Recalculate pitch result with smoothed frequency using configured base A
            pitch.freqHz = smoothedPitchHz_;
            double baseA = transform_->baseAFrequency();
            double midiExact = 69.0 + 12.0 * std::log2(smoothedPitchHz_ / baseA);
            int midiNote = static_cast<int>(std::round(midiExact));
            midiNote = std::clamp(midiNote, 0, 127);
            pitch.midiNote = midiNote;
            pitch.cents = (midiExact - std::round(midiExact)) * 100.0;
            
            static const char* NOTE_NAMES[] = {"C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"};
            int noteIndex = ((midiNote % 12) + 12) % 12;
            int octave = (midiNote / 12) - 1;
            static char noteBuf[8];
            std::snprintf(noteBuf, sizeof(noteBuf), "%s%d", NOTE_NAMES[noteIndex], octave);
            pitch.noteName = noteBuf;
        }
        frameStats_.pitch = pitch;
    } else {
        frameStats_.pitch = {};
    }

    if (!useDirectGpuPipeline()
        && frameStats_.pitch.confidence > 0.2
        && frameStats_.pitch.midiNote >= 0
        && frameStats_.pitch.freqHz > 0.0) {
        displayPitchHz_ = frameStats_.pitch.freqHz;
        displayPitchCents_ = frameStats_.pitch.cents;
        displayPitchConfidence_ = std::clamp(frameStats_.pitch.confidence, 0.0, 1.0);
        displayPitchMidiNote_ = frameStats_.pitch.midiNote;
        displayPitchNoteName_ = QString::fromStdString(frameStats_.pitch.noteName);
        lastPitchUpdateMs_ = now;
    }

    if (auto* glCanvas = dynamic_cast<GlSpectrogramCanvas*>(canvas_)) {
        glCanvas->requestRepaint();
    } else {
        canvas_->update();
    }
}

SpectrogramWidget::RGB SpectrogramWidget::colormap(float amplitude) const
{
    float t = visualLevel(amplitude);
    if (t <= 0.0f) return {0, 0, 0};

    uint8_t r, g, b;
    if (t < 0.15f) {
        float s = t / 0.15f;
        r = 0; g = 0; b = static_cast<uint8_t>(s * 200);
    } else if (t < 0.35f) {
        float s = (t - 0.15f) / 0.2f;
        r = 0; g = static_cast<uint8_t>(s * 255); b = static_cast<uint8_t>(200 + s * 55);
    } else if (t < 0.55f) {
        float s = (t - 0.35f) / 0.2f;
        r = 0; g = 255; b = static_cast<uint8_t>(255 * (1 - s));
    } else if (t < 0.75f) {
        float s = (t - 0.55f) / 0.2f;
        r = static_cast<uint8_t>(s * 255); g = 255; b = 0;
    } else {
        float s = (t - 0.75f) / 0.25f;
        r = 255; g = static_cast<uint8_t>(255 * (1 - s * 0.6f)); b = static_cast<uint8_t>(s * 180);
    }
    return {r, g, b};
}

float SpectrogramWidget::visualLevel(float amplitude) const
{
    float logMax = std::log(1.0f + maxAmplitude_ * gain_);
    float t = logMax > 0 ? std::log(1.0f + amplitude * gain_) / logMax : 0;
    t = std::clamp(t, 0.0f, 1.0f);

    if (t <= floor_) return 0.0f;
    t = (t - floor_) / std::max(1e-6f, 1.0f - floor_);
    return applyToneCurve(std::clamp(t, 0.0f, 1.0f));
}

float SpectrogramWidget::applyToneCurve(float t) const
{
    t = std::clamp(t, 0.0f, 1.0f);
    switch (toneCurveMode_) {
    case ToneCurveMode::PowerGamma:
        return std::pow(t, gamma_);
    case ToneCurveMode::Smoothstep: {
        float shaped = std::pow(t, std::max(0.05f, gamma_));
        return shaped * shaped * (3.0f - 2.0f * shaped);
    }
    case ToneCurveMode::Sigmoid: {
        float steepness = std::clamp(2.0f + (1.0f - gamma_) * 10.0f, 2.0f, 14.0f);
        float x = (t - 0.5f) * steepness;
        float lo = 1.0f / (1.0f + std::exp(steepness * 0.5f));
        float hi = 1.0f / (1.0f + std::exp(-steepness * 0.5f));
        float y = 1.0f / (1.0f + std::exp(-x));
        return std::clamp((y - lo) / std::max(1e-6f, hi - lo), 0.0f, 1.0f);
    }
    case ToneCurveMode::CustomCurve:
        break;
    }

    if (customToneCurve_.size() < 2) return t;
    if (t <= customToneCurve_.front().x()) return std::clamp(static_cast<float>(customToneCurve_.front().y()), 0.0f, 1.0f);
    if (t >= customToneCurve_.back().x()) return std::clamp(static_cast<float>(customToneCurve_.back().y()), 0.0f, 1.0f);
    for (size_t i = 1; i < customToneCurve_.size(); ++i) {
        const QPointF& a = customToneCurve_[i - 1];
        const QPointF& b = customToneCurve_[i];
        if (t > b.x()) continue;
        double span = std::max(1e-9, b.x() - a.x());
        double localT = (t - a.x()) / span;
        double y = a.y() + (b.y() - a.y()) * localT;
        return std::clamp(static_cast<float>(y), 0.0f, 1.0f);
    }
    return t;
}

void SpectrogramWidget::paintDecorations(QPainter& p, const QSize& canvasSize)
{
    int nb = transform_->numBins();
    if (nb < 1) return;

    QRect spectRect = spectrogramRect(canvasSize);
    QRect histRect = histogramRect(canvasSize);
    
    // Y-axis column rect (frequency labels area)
    QRect yAxisRect(0, spectRect.top(), Y_AXIS_WIDTH - 5, spectRect.height());

    p.setRenderHint(QPainter::TextAntialiasing, true);
    
    // Draw Y-axis background
    p.fillRect(yAxisRect.adjusted(-2, 0, 2, 0), QColor(12, 12, 20));
    
    // Draw separator line between Y-axis and spectrogram
    p.setPen(QColor(50, 50, 70));
    p.drawLine(Y_AXIS_WIDTH - 3, spectRect.top(), Y_AXIS_WIDTH - 3, spectRect.bottom());

    double fMin = transform_->binFreqHz(0);
    double fMax = transform_->binFreqHz(nb - 1);
    double logMin = std::log(fMin);
    double logRange = std::max(0.0001, std::log(fMax) - logMin);

    // Draw frequency labels in the Y-axis column
    QFont labelFont = p.font();
    labelFont.setPixelSize(10);
    p.setFont(labelFont);
    
    const double labelFreqs[] = {50, 100, 200, 440, 500, 1000, 2000, 3000, 5000, 8000, 10000};
    for (double f : labelFreqs) {
        if (f < fMin || f > fMax) continue;
        double logPos = (std::log(f) - logMin) / logRange;
        int y = binToY(nb, spectRect, logPos * (nb - 1));
        QString label = f >= 1000 ? QString("%1k").arg(f / 1000.0, 0, 'f', 1)
                                  : QString::number(static_cast<int>(f));
        
        if (gridVisible_) {
            // Draw horizontal grid line across spectrogram only
            p.setPen(QColor(40, 40, 60));
            p.drawLine(spectRect.left(), y, spectRect.right(), y);
        }
        
        // Draw tick mark on the separator line
        p.setPen(QColor(100, 100, 130));
        p.drawLine(Y_AXIS_WIDTH - 6, y, Y_AXIS_WIDTH - 3, y);
        
        // Draw label right-aligned in Y-axis column
        QFontMetrics fm(labelFont);
        int textW = fm.horizontalAdvance(label);
        p.setPen(QColor(200, 200, 230));  // Brighter text
        p.drawText(Y_AXIS_WIDTH - 8 - textW, y + 4, label);
    }

    // Histogram background
    p.fillRect(histRect.adjusted(-1, 0, 1, 0), QColor(12, 12, 20));
    p.setPen(QColor(30, 30, 50));
    p.drawLine(histRect.left() - 1, spectRect.top(), histRect.left() - 1, spectRect.bottom());

    auto stats = transform_->getStats();
    QFont monoFont("Menlo", 9);
    p.setFont(monoFont);
    bool directGpu = useDirectGpuPipeline();

    QStringList lines;
    lines << QString("FPS: %1").arg(frameStats_.fps, 0, 'f', 0);
    lines << QString("Span: %1 s").arg(displaySeconds_, 0, 'f', 1);
    lines << QString("Mode: %1/%2")
             .arg(hardwareAccelerationEnabled_ ? "gpu" : "cpu")
             .arg(LoiaconoRolling::computeModeName(transform_->activeComputeMode()));
    lines << QString("Algo: %1").arg(LoiaconoRolling::algorithmModeName(transform_->algorithmMode()));
    lines << QString("Bins: %1 x%2").arg(stats.currentBins).arg(stats.currentMultiple);
    lines << QString("Load: %1%  %2 kS/s")
             .arg(stats.cpuLoadPercent, 0, 'f', 1)
             .arg(stats.samplesPerSecond / 1000.0, 0, 'f', 1);
    if (!directGpu) {
        lines << QString("Peak: %1 Hz").arg(frameStats_.peakHz, 0, 'f', 0);
    }

    int lineH = 13;
    int boxW = 148;
    int boxH = lines.size() * lineH + 6;
    int boxX = spectRect.right() - boxW - 3;
    int boxY = 4;

    p.fillRect(boxX, boxY, boxW, boxH, QColor(0, 0, 0, 180));
    p.setPen(QColor(30, 30, 50));
    p.drawRect(boxX, boxY, boxW, boxH);

    p.setPen(QColor(160, 200, 255));
    for (int i = 0; i < lines.size(); i++) {
        p.drawText(boxX + 4, boxY + 12 + i * lineH, lines[i]);
    }

    const qint64 nowMs = QDateTime::currentMSecsSinceEpoch();
    const qint64 pitchAgeMs = lastPitchUpdateMs_ > 0
        ? std::max<qint64>(0, nowMs - lastPitchUpdateMs_)
        : (PITCH_PERSIST_MS + 1);
    const double pitchFade = std::clamp(
        1.0 - (static_cast<double>(pitchAgeMs) / static_cast<double>(PITCH_PERSIST_MS)),
        0.0, 1.0);
    const double pitchStrength = std::clamp(displayPitchConfidence_ * pitchFade, 0.0, 1.0);
    const bool showPitchDial = displayPitchMidiNote_ >= 0 && pitchStrength > 0.03;

    int dialSize = 84;
    int dialX = boxX + boxW - dialSize - 6;
    int dialY = boxY + boxH + 6;
    if (dialY + dialSize + 36 > spectRect.bottom()) {
        dialY = boxY + 2;
        dialX = std::max(spectRect.left() + 6, boxX - dialSize - 8);
    }
    dialX = std::clamp(dialX, spectRect.left() + 4, spectRect.right() - dialSize - 4);
    dialY = std::clamp(dialY, spectRect.top() + 4, spectRect.bottom() - dialSize - 30);

    QRect dialPanel(dialX - 6, dialY - 12, dialSize + 12, dialSize + 36);
    p.fillRect(dialPanel, QColor(0, 0, 0, 175));
    p.setPen(QColor(30, 30, 50));
    p.drawRect(dialPanel);

    QRectF dialRect(dialX, dialY, dialSize, dialSize);
    auto centsToDegrees = [](double cents) {
        const double clamped = std::clamp(cents, -50.0, 50.0);
        return 225.0 - ((clamped + 50.0) / 100.0) * 270.0;
    };

    p.setPen(QPen(QColor(70, 80, 95), 4));
    p.drawArc(dialRect, 225 * 16, -270 * 16);

    const int ringAlpha = static_cast<int>(70 + 185 * pitchStrength);
    p.setPen(QPen(QColor(80, 210, 120, ringAlpha), 4));
    const int tuneStart = static_cast<int>(std::round(centsToDegrees(-5.0) * 16.0));
    const int tuneSpan = static_cast<int>(std::round((centsToDegrees(5.0) - centsToDegrees(-5.0)) * 16.0));
    p.drawArc(dialRect, tuneStart, tuneSpan);

    const double shownCents = showPitchDial ? std::clamp(displayPitchCents_, -50.0, 50.0) : 0.0;
    const double needleDeg = centsToDegrees(shownCents);
    constexpr double kPi = 3.14159265358979323846;
    const double needleRad = needleDeg * (kPi / 180.0);
    const QPointF center = dialRect.center();
    const double radius = (dialSize * 0.5) - 8.0;
    const QPointF needleEnd(center.x() + radius * std::cos(needleRad),
                            center.y() - radius * std::sin(needleRad));

    QColor needleColor(150, 160, 175);
    if (showPitchDial) {
        const double absCents = std::abs(shownCents);
        if (absCents <= 5.0) {
            needleColor = QColor(80, 220, 120, ringAlpha);
        } else if (absCents <= 15.0) {
            needleColor = QColor(240, 210, 80, ringAlpha);
        } else {
            needleColor = QColor(255, 120, 90, ringAlpha);
        }
    }
    p.setPen(QPen(needleColor, 3));
    p.drawLine(center, needleEnd);
    p.setBrush(QColor(210, 220, 235, 220));
    p.setPen(Qt::NoPen);
    p.drawEllipse(center, 3, 3);

    QFont pitchLabelFont("Menlo", 9, QFont::Bold);
    p.setFont(pitchLabelFont);
    p.setPen(QColor(185, 210, 235));
    p.drawText(QRect(dialX, dialY - 10, dialSize, 10), Qt::AlignCenter, "Pitch");

    QString noteText = showPitchDial ? displayPitchNoteName_ : "--";
    QString centsText = showPitchDial
        ? QString("%1%2 c").arg(shownCents >= 0 ? "+" : "").arg(static_cast<int>(std::lround(shownCents)))
        : "-- c";
    QString freqText = showPitchDial
        ? QString("%1 Hz").arg(displayPitchHz_, 0, 'f', 1)
        : "-- Hz";

    p.setPen(QColor(225, 235, 255, 235));
    p.drawText(QRect(dialX, dialY + 24, dialSize, 16), Qt::AlignCenter, noteText);

    QFont pitchInfoFont("Menlo", 8);
    p.setFont(pitchInfoFont);
    p.setPen(QColor(180, 200, 225, 220));
    p.drawText(QRect(dialX, dialY + dialSize - 22, dialSize, 12), Qt::AlignCenter, centsText);
    p.drawText(QRect(dialX, dialY + dialSize - 10, dialSize, 12), Qt::AlignCenter, freqText);

    QRect confidenceBar(dialX + 6, dialY + dialSize + 16, dialSize - 12, 6);
    p.setPen(QColor(70, 80, 95));
    p.setBrush(QColor(20, 20, 30, 220));
    p.drawRect(confidenceBar);
    int confidenceWidth = static_cast<int>(std::round((confidenceBar.width() - 2) * std::clamp(pitchStrength, 0.0, 1.0)));
    if (confidenceWidth > 0) {
        p.fillRect(confidenceBar.adjusted(1, 1, -1 - ((confidenceBar.width() - 2) - confidenceWidth), -1),
                   QColor(80, 220, 140, 210));
    }

    if (!directGpu) {
        int barH = 8;
        int barY = std::max(4, spectRect.bottom() - barH - 1);
        int barX = Y_AXIS_WIDTH + 4;  // Move to right of Y-axis column
        int barW2 = 200;
        for (int i = 0; i < barW2; i++) {
            float t = static_cast<float>(i) / barW2;
            float amp = t * maxAmplitude_;
            auto [cr, cg, cb] = colormap(amp);
            p.setPen(QColor(cr, cg, cb));
            p.drawLine(barX + i, barY, barX + i, barY + barH);
        }
        p.setPen(QColor(80, 80, 100));
        p.drawRect(barX, barY, barW2, barH);

        QFont smallFont = p.font();
        smallFont.setPixelSize(10);
        p.setFont(smallFont);
        p.setPen(QColor(100, 100, 130));
        p.drawText(barX, barY - 2, QString("gain:%1 gamma:%2 floor:%3 curve:%4")
                   .arg(gain_, 0, 'f', 1).arg(gamma_, 0, 'f', 2).arg(floor_, 0, 'f', 2)
                   .arg(toneCurveModeName(toneCurveMode_)));
    }

    int axisY = spectRect.bottom() + 1;
    // X-axis background (dark) - covers full width including Y-axis column
    p.fillRect(0, axisY, canvasSize.width(), AXIS_HEIGHT, QColor(12, 12, 20));
    p.setPen(QColor(35, 35, 55));
    p.drawLine(0, axisY, histRect.right(), axisY);

    p.setPen(QColor(120, 120, 150));
    p.drawText(Y_AXIS_WIDTH, axisY + 12, "Time (s)");

    int maxWholeSeconds = std::max(1, static_cast<int>(std::floor(displaySeconds_)));
    for (int second = 0; second <= maxWholeSeconds; ++second) {
        double ageSeconds = static_cast<double>(second);
        int x = spectRect.width() - 1 - static_cast<int>((ageSeconds / displaySeconds_) * (spectRect.width() - 1));
        x = std::clamp(x, spectRect.left(), spectRect.right());
        if (gridVisible_) {
            p.setPen(QColor(40, 40, 60));
            p.drawLine(x, spectRect.top(), x, spectRect.bottom());
        }
        p.setPen(QColor(120, 120, 150));
        p.drawLine(x, axisY, x, axisY + 4);
        QString label = second == 0 ? "0" : QString("-%1").arg(second);
        p.drawText(std::max(0, x - 8), axisY + 14, label);
    }

    if (bufferEdgeMarkersVisible_ && audioBufferFrames_ > 0) {
        const uint64_t totalSamples = stats.totalSamples;
        const double displaySamples = displaySeconds_ * transform_->sampleRate();
        const uint64_t maxVisibleAgeSamples = static_cast<uint64_t>(std::ceil(std::max(1.0, displaySamples)));
        const uint64_t firstVisibleSample = totalSamples > maxVisibleAgeSamples
            ? (totalSamples - maxVisibleAgeSamples)
            : 0;
        uint64_t firstBoundary = ((firstVisibleSample + audioBufferFrames_ - 1) / audioBufferFrames_) * audioBufferFrames_;
        p.setPen(QColor(255, 180, 70, 140));
        for (uint64_t boundary = firstBoundary; boundary <= totalSamples; boundary += audioBufferFrames_) {
            const uint64_t ageSamples = totalSamples - boundary;
            if (static_cast<double>(ageSamples) > displaySamples) continue;
            int x = spectRect.right() - static_cast<int>(
                std::lround((static_cast<double>(ageSamples) / std::max(1.0, displaySamples)) * (spectRect.width() - 1)));
            x = std::clamp(x, spectRect.left(), spectRect.right());
            p.drawLine(x, spectRect.top(), x, spectRect.bottom());
            p.drawLine(x, axisY, x, axisY + 6);
        }
    }
}

void SpectrogramWidget::paintContent(QPainter& p, const QSize& canvasSize)
{
    int nb = transform_->numBins();
    if (nb < 1) return;
    const int spectrumBins = static_cast<int>(spectrum_.size());
    const int usableBins = std::min(nb, spectrumBins);

    QRect spectRect = spectrogramRect(canvasSize);
    QRect histRect = histogramRect(canvasSize);

    // Background for canvas area
    p.fillRect(QRect(QPoint(0, 0), canvasSize), QColor(10, 10, 16));
    p.drawImage(spectRect, image_);
    // Histogram background (dark)
    p.fillRect(histRect.adjusted(-1, 0, 1, 0), QColor(12, 12, 20));

    if (usableBins > 0 && maxAmplitude_ > 0) {
        for (int fi = 0; fi < usableBins; fi++) {
            float t = visualLevel(spectrum_[fi]);
            int barW = static_cast<int>(t * histRect.width());
            if (barW < 1) continue;

            int yTop = binToY(nb, spectRect, std::min(nb - 1.0, fi + 0.5));
            int yBottom = binToY(nb, spectRect, std::max(0.0, fi - 0.5));
            int row = std::min(yTop, yBottom);
            int barH = std::max(1, std::abs(yBottom - yTop) + 1);
            auto [r, g, b] = colormap(spectrum_[fi]);
            p.fillRect(histRect.left(), row, barW, barH, QColor(r, g, b));
        }
    }

    paintDecorations(p, canvasSize);
}

QImage SpectrogramWidget::renderToImage() const
{
    QImage img(size(), QImage::Format_RGB32);
    img.fill(Qt::black);
    QPainter p(&img);
    const_cast<SpectrogramWidget*>(this)->paintContent(p, img.size());
    return img;
}
