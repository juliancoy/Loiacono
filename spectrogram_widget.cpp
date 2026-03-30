#include "spectrogram_widget.h"
#include "gl_spectrogram_canvas.h"

#include <QDateTime>
#include <QHBoxLayout>
#include <QPainter>
#include <QPaintEvent>
#include <QResizeEvent>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <cstring>

class SpectrogramWidget::RasterCanvas : public QWidget {
public:
    explicit RasterCanvas(SpectrogramWidget* owner) : QWidget(owner), owner_(owner) {}

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

private:
    SpectrogramWidget* owner_;
};

SpectrogramWidget::SpectrogramWidget(LoiaconoRolling* transform, QWidget* parent)
    : QWidget(parent), transform_(transform)
{
    setMinimumSize(600, 300);

    auto* layout = new QHBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    replaceCanvas();

    image_ = QImage(1, 1, QImage::Format_RGB32);
    image_.fill(Qt::black);
    lastFpsTime_ = QDateTime::currentMSecsSinceEpoch();

    connect(&timer_, &QTimer::timeout, this, &SpectrogramWidget::tick);
    timer_.start(0);
}

void SpectrogramWidget::setHardwareAccelerationEnabled(bool enabled)
{
    if (hardwareAccelerationEnabled_ == enabled) return;
    hardwareAccelerationEnabled_ = enabled;
    replaceCanvas();
}

void SpectrogramWidget::setDisplayedTimeSeconds(double seconds)
{
    displaySeconds_ = std::max(0.5, seconds);
    lastColumnSampleCount_ = transform_->getStats().totalSamples;
    pendingColumnFraction_ = 0.0;
    pendingGpuColumns_ = 0;
    update();
}

void SpectrogramWidget::resetHistory()
{
    image_.fill(Qt::black);
    lastColumnSampleCount_ = transform_->getStats().totalSamples;
    pendingColumnFraction_ = 0.0;
    pendingGpuColumns_ = 0;
    historyRevision_++;
    if (canvas_) canvas_->update();
}

void SpectrogramWidget::onCanvasResized()
{
    QRect spectRect = spectrogramRect(canvas_->size());
    
    // Resize the image to match new canvas size
    if (image_.width() != spectRect.width() || image_.height() != spectRect.height()) {
        QImage newImage(spectRect.width(), spectRect.height(), QImage::Format_RGB32);
        newImage.fill(Qt::black);
        image_ = newImage;
    }
    
    lastColumnSampleCount_ = transform_->getStats().totalSamples;
    pendingColumnFraction_ = 0.0;
    pendingGpuColumns_ = 0;
    historyRevision_++;
    
    if (canvas_) canvas_->update();
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

    layout->addWidget(canvas_);
    canvas_->setMinimumSize(600, 300);
    canvas_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
}

QRect SpectrogramWidget::spectrogramRect(const QSize& canvasSize) const
{
    return QRect(0, 0, std::max(1, canvasSize.width() - HISTOGRAM_WIDTH), std::max(1, canvasSize.height() - AXIS_HEIGHT));
}

QRect SpectrogramWidget::histogramRect(const QSize& canvasSize) const
{
    QRect spectRect = spectrogramRect(canvasSize);
    return QRect(spectRect.right() + 2, spectRect.top(), HISTOGRAM_WIDTH - 2, spectRect.height());
}

int SpectrogramWidget::binToY(int numBins, const QRect& rect, double binIndex) const
{
    if (numBins <= 1) return rect.bottom();
    double t = std::clamp(binIndex / (numBins - 1.0), 0.0, 1.0);
    return rect.bottom() - static_cast<int>(std::lround(t * (rect.height() - 1)));
}

bool SpectrogramWidget::useDirectGpuPipeline() const
{
    return hardwareAccelerationEnabled_ &&
           transform_->activeComputeMode() == LoiaconoRolling::ComputeMode::GpuCompute;
}

void SpectrogramWidget::handleWheelZoom(const QPoint& position, int angleDeltaY, const QSize& canvasSize)
{
    if (angleDeltaY == 0) return;

    QRect spectRect = spectrogramRect(canvasSize);
    QRect histRect = histogramRect(canvasSize);
    QRect xAxisRect(spectRect.left(), spectRect.bottom() + 1, spectRect.width(), AXIS_HEIGHT);
    QRect yAxisHotRect(spectRect.left(), spectRect.top(), std::min(56, spectRect.width()), spectRect.height());

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

    Q_UNUSED(histRect);
}

void SpectrogramWidget::tick()
{
    int nb = transform_->numBins();
    if (nb < 1 || !canvas_) return;

    QRect spectRect = spectrogramRect(canvas_->size());

    if (image_.width() != spectRect.width() || image_.height() != spectRect.height()) {
        // Resize happened - onCanvasResized() will handle the reset
        // Just resize the image here, reset is done in resize handler
        QImage newImage(spectRect.width(), spectRect.height(), QImage::Format_RGB32);
        newImage.fill(Qt::black);
        image_ = newImage;
        lastColumnSampleCount_ = transform_->getStats().totalSamples;
        pendingColumnFraction_ = 0.0;
        pendingGpuColumns_ = 0;
        historyRevision_++;
    }

    if (!useDirectGpuPipeline()) {
        transform_->getSpectrum(spectrum_);
    }

    float currentMax = frameStats_.peakAmp;
    int peakIdx = 0;
    if (!useDirectGpuPipeline()) {
        currentMax = 0;
        for (int i = 0; i < static_cast<int>(spectrum_.size()); i++) {
            if (spectrum_[i] > currentMax) {
                currentMax = spectrum_[i];
                peakIdx = i;
            }
        }
    }
    if (currentMax > maxAmplitude_) {
        maxAmplitude_ = currentMax;
    } else {
        maxAmplitude_ = maxAmplitude_ * 0.997f + currentMax * 0.003f;
    }
    if (maxAmplitude_ < 0.01f) maxAmplitude_ = 0.01f;

    int w = image_.width();
    int h = image_.height();
    auto runtimeStats = transform_->getStats();
    if (lastColumnSampleCount_ == 0) {
        lastColumnSampleCount_ = runtimeStats.totalSamples;
    }
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
        for (int fi = 0; fi < nb && fi < h; fi++) {
            int row = h - 1 - fi;
            auto [r, g, b] = colormap(spectrum_[fi]);
            auto* line = reinterpret_cast<QRgb*>(image_.scanLine(row));
            for (int x = w - columnsToAdvance; x < w; x++) {
                line[x] = qRgb(r, g, b);
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
    }

    if (auto* glCanvas = dynamic_cast<GlSpectrogramCanvas*>(canvas_)) {
        glCanvas->requestRepaint();
    } else {
        canvas_->update();
    }
}

SpectrogramWidget::RGB SpectrogramWidget::colormap(float amplitude) const
{
    float logMax = std::log(1.0f + maxAmplitude_ * gain_);
    float t = logMax > 0 ? std::log(1.0f + amplitude * gain_) / logMax : 0;
    t = std::clamp(t, 0.0f, 1.0f);

    if (t < floor_) return {0, 0, 0};

    t = (t - floor_) / (1.0f - floor_);
    t = std::pow(t, gamma_);

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

void SpectrogramWidget::paintDecorations(QPainter& p, const QSize& canvasSize)
{
    int nb = transform_->numBins();
    if (nb < 1) return;

    QRect spectRect = spectrogramRect(canvasSize);
    QRect histRect = histogramRect(canvasSize);

    p.setRenderHint(QPainter::TextAntialiasing, true);
    p.setPen(QColor(140, 140, 180));
    QFont smallFont = p.font();
    smallFont.setPixelSize(10);
    p.setFont(smallFont);

    double fMin = transform_->binFreqHz(0);
    double fMax = transform_->binFreqHz(nb - 1);
    double logMin = std::log(fMin);
    double logRange = std::max(0.0001, std::log(fMax) - logMin);

    const double labelFreqs[] = {50, 100, 200, 440, 500, 1000, 2000, 3000, 5000, 8000, 10000};
    for (double f : labelFreqs) {
        if (f < fMin || f > fMax) continue;
        double logPos = (std::log(f) - logMin) / logRange;
        int y = binToY(nb, spectRect, logPos * (nb - 1));
        QString label = f >= 1000 ? QString("%1k").arg(f / 1000.0, 0, 'g', 3)
                                  : QString::number(static_cast<int>(f));
        p.setPen(QColor(60, 60, 80));
        p.drawLine(spectRect.left(), y, histRect.right(), y);
        p.setPen(QColor(140, 140, 180));
        p.drawText(4, std::max(10, y - 2), label);
    }

    p.fillRect(histRect.adjusted(-1, 0, 1, 0), QColor(0, 0, 0, 0));
    p.setPen(QColor(30, 30, 50));
    p.drawLine(histRect.left() - 1, spectRect.top(), histRect.left() - 1, spectRect.bottom());

    auto stats = transform_->getStats();
    QFont monoFont("Menlo", 9);
    p.setFont(monoFont);
    bool directGpu = useDirectGpuPipeline();

    QStringList lines;
    lines << QString("FPS: %1").arg(frameStats_.fps, 0, 'f', 0);
    lines << QString("Displayed: %1 s").arg(displaySeconds_, 0, 'f', 1);
    lines << QString("Render: %1").arg(hardwareAccelerationEnabled_ ? "GPU" : "CPU");
    lines << QString("CPU thr: %1").arg(transform_->cpuThreads());
    lines << QString("Compute: %1").arg(LoiaconoRolling::computeModeName(transform_->activeComputeMode()));
    lines << QString("Bins: %1 x%2").arg(stats.currentBins).arg(stats.currentMultiple);
    lines << QString("CPU: %1%").arg(stats.cpuLoadPercent, 0, 'f', 1);
    lines << QString("%1 kS/s").arg(stats.samplesPerSecond / 1000.0, 0, 'f', 1);
    if (!directGpu) {
        lines.insert(1, QString("Peak: %1 Hz").arg(frameStats_.peakHz, 0, 'f', 0));
    }

    int lineH = 13;
    int boxW = 136;
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

    if (!directGpu) {
        int barH = 8;
        int barY = std::max(4, spectRect.bottom() - barH - 1);
        int barX = 4;
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

        p.setFont(smallFont);
        p.setPen(QColor(100, 100, 130));
        p.drawText(barX, barY - 2, QString("gain:%1 gamma:%2 floor:%3")
                   .arg(gain_, 0, 'f', 1).arg(gamma_, 0, 'f', 2).arg(floor_, 0, 'f', 2));
    }

    int axisY = spectRect.bottom() + 1;
    p.fillRect(0, axisY, spectRect.width(), AXIS_HEIGHT, QColor(12, 12, 20));
    p.fillRect(histRect.left() - 1, axisY, HISTOGRAM_WIDTH, AXIS_HEIGHT, QColor(12, 12, 20));
    p.setPen(QColor(35, 35, 55));
    p.drawLine(0, axisY, histRect.right(), axisY);

    p.setPen(QColor(120, 120, 150));
    p.drawText(4, axisY + 12, "Time (s)");

    int maxWholeSeconds = std::max(1, static_cast<int>(std::floor(displaySeconds_)));
    for (int second = 0; second <= maxWholeSeconds; ++second) {
        double ageSeconds = static_cast<double>(second);
        int x = spectRect.width() - 1 - static_cast<int>((ageSeconds / displaySeconds_) * (spectRect.width() - 1));
        x = std::clamp(x, spectRect.left(), spectRect.right());
        p.setPen(QColor(40, 40, 60));
        p.drawLine(x, spectRect.top(), x, spectRect.bottom());
        p.drawLine(x, axisY, x, axisY + 4);
        p.setPen(QColor(120, 120, 150));
        QString label = second == 0 ? "0" : QString("-%1").arg(second);
        p.drawText(std::max(0, x - 8), axisY + 14, label);
    }
}

void SpectrogramWidget::paintContent(QPainter& p, const QSize& canvasSize)
{
    int nb = transform_->numBins();
    if (nb < 1) return;

    QRect spectRect = spectrogramRect(canvasSize);
    QRect histRect = histogramRect(canvasSize);

    p.fillRect(QRect(QPoint(0, 0), canvasSize), QColor(10, 10, 16));
    p.drawImage(spectRect, image_);
    p.fillRect(histRect.adjusted(-1, 0, 1, 0), QColor(12, 12, 20));

    if (!spectrum_.empty() && maxAmplitude_ > 0) {
        float logMax = std::log(1.0f + maxAmplitude_ * gain_);
        for (int fi = 0; fi < nb; fi++) {
            float t = logMax > 0 ? std::log(1.0f + spectrum_[fi] * gain_) / logMax : 0;
            t = std::clamp(t, 0.0f, 1.0f);
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
