#include "spectrogram_widget.h"
#include <QPainter>
#include <QPaintEvent>
#include <QElapsedTimer>
#include <QDateTime>
#include <cmath>

SpectrogramWidget::SpectrogramWidget(LoiaconoRolling* transform, QWidget* parent)
    : QWidget(parent), transform_(transform)
{
    setMinimumSize(600, 300);
    image_ = QImage(1, 1, QImage::Format_RGB32);
    image_.fill(Qt::black);
    lastFpsTime_ = QDateTime::currentMSecsSinceEpoch();

    connect(&timer_, &QTimer::timeout, this, &SpectrogramWidget::tick);
    timer_.start(0); // full speed —
}

void SpectrogramWidget::tick()
{
    int nb = transform_->numBins();
    if (nb < 1) return;

    int spectW = std::max(1, width() - HISTOGRAM_WIDTH);
    int spectH = nb;

    // Resize backing image if needed
    if (image_.width() != spectW || image_.height() != spectH) {
        QImage newImage(spectW, spectH, QImage::Format_RGB32);
        newImage.fill(Qt::black);
        image_ = newImage;
    }

    // Get spectrum
    transform_->getSpectrum(spectrum_);

    // Auto-adjust max amplitude
    float currentMax = 0;
    int peakIdx = 0;
    for (int i = 0; i < (int)spectrum_.size(); i++) {
        if (spectrum_[i] > currentMax) {
            currentMax = spectrum_[i];
            peakIdx = i;
        }
    }
    if (currentMax > maxAmplitude_) {
        maxAmplitude_ = currentMax;
    } else {
        maxAmplitude_ = maxAmplitude_ * 0.997f + currentMax * 0.003f;
    }
    if (maxAmplitude_ < 0.01f) maxAmplitude_ = 0.01f;

    // Scroll spectrogram left by 1 pixel
    int w = image_.width();
    int h = image_.height();
    for (int y = 0; y < h; y++) {
        auto* line = reinterpret_cast<QRgb*>(image_.scanLine(y));
        std::memmove(line, line + 1, (w - 1) * sizeof(QRgb));
    }

    // Write new column
    for (int fi = 0; fi < nb && fi < h; fi++) {
        int row = h - 1 - fi;
        auto [r, g, b] = colormap(spectrum_[fi]);
        auto* line = reinterpret_cast<QRgb*>(image_.scanLine(row));
        line[w - 1] = qRgb(r, g, b);
    }

    // FPS tracking
    frameCount_++;
    qint64 now = QDateTime::currentMSecsSinceEpoch();
    if (now - lastFpsTime_ >= 500) {
        frameStats_.fps = frameCount_ * 1000.0 / (now - lastFpsTime_);
        frameCount_ = 0;
        lastFpsTime_ = now;
    }
    frameStats_.peakHz = nb > 0 ? transform_->binFreqHz(peakIdx) : 0;
    frameStats_.peakAmp = currentMax;
    frameStats_.maxAmp = maxAmplitude_;

    update();
}

SpectrogramWidget::RGB SpectrogramWidget::colormap(float amplitude) const
{
    // Apply gain, log compression, floor, gamma
    float logMax = std::log(1.0f + maxAmplitude_ * gain_);
    float t = logMax > 0 ? std::log(1.0f + amplitude * gain_) / logMax : 0;
    t = std::clamp(t, 0.0f, 1.0f);

    if (t < floor_) return {0, 0, 0};

    // Remap above floor to 0-1
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

void SpectrogramWidget::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    int nb = transform_->numBins();
    if (nb < 1) return;

    int spectW = width() - HISTOGRAM_WIDTH;
    int spectH = height();

    // ── Spectrogram ──
    p.drawImage(QRect(0, 0, spectW, spectH), image_);

    // Frequency labels on spectrogram
    p.setPen(QColor(140, 140, 180));
    QFont smallFont = p.font();
    smallFont.setPixelSize(10);
    p.setFont(smallFont);

    double fMin = transform_->binFreqHz(0);
    double fMax = transform_->binFreqHz(nb - 1);
    double logMin = std::log(fMin);
    double logRange = std::log(fMax) - logMin;

    const double labelFreqs[] = {50,100,200,440,500,1000,2000,3000,5000,8000,10000};
    for (double f : labelFreqs) {
        if (f < fMin || f > fMax) continue;
        double logPos = (std::log(f) - logMin) / logRange;
        int y = spectH - static_cast<int>(logPos * spectH);
        QString label = f >= 1000 ? QString("%1k").arg(f / 1000.0, 0, 'g', 3)
                                  : QString::number(static_cast<int>(f));
        p.setPen(QColor(60, 60, 80));
        p.drawLine(0, y, spectW, y);
        p.setPen(QColor(140, 140, 180));
        p.drawText(4, y - 2, label);
    }

    // ── Histogram (right panel) ──
    int histX = spectW + 1;
    int histW = HISTOGRAM_WIDTH - 2;

    // Background
    p.fillRect(histX - 1, 0, HISTOGRAM_WIDTH, spectH, QColor(12, 12, 20));
    p.setPen(QColor(30, 30, 50));
    p.drawLine(histX - 1, 0, histX - 1, spectH);

    // Draw bars
    if (!spectrum_.empty() && maxAmplitude_ > 0) {
        float logMax = std::log(1.0f + maxAmplitude_ * gain_);
        for (int fi = 0; fi < nb; fi++) {
            float t = logMax > 0 ? std::log(1.0f + spectrum_[fi] * gain_) / logMax : 0;
            t = std::clamp(t, 0.0f, 1.0f);
            int barW = static_cast<int>(t * histW);
            if (barW < 1) continue;

            int row = spectH - 1 - static_cast<int>((float)fi / nb * spectH);
            auto [r, g, b] = colormap(spectrum_[fi]);
            p.fillRect(histX, row, barW, std::max(1, spectH / nb), QColor(r, g, b));
        }
    }

    // ── Runtime stats overlay (top-right of spectrogram) ──
    auto stats = transform_->getStats();
    QFont monoFont("Menlo", 9);
    p.setFont(monoFont);

    QStringList lines;
    lines << QString("FPS: %1").arg(frameStats_.fps, 0, 'f', 0);
    lines << QString("Peak: %1 Hz").arg(frameStats_.peakHz, 0, 'f', 0);
    lines << QString("Amp: %1").arg(frameStats_.peakAmp, 0, 'f', 2);
    lines << QString("Bins: %1 x%2").arg(stats.currentBins).arg(stats.currentMultiple);
    lines << QString("CPU: %1%").arg(stats.cpuLoadPercent, 0, 'f', 1);
    lines << QString("%1 kS/s").arg(stats.samplesPerSecond / 1000.0, 0, 'f', 1);
    lines << QString("Chunk: %1 us").arg(stats.avgChunkMicros, 0, 'f', 0);

    int lineH = 13;
    int boxW = 130;
    int boxH = lines.size() * lineH + 6;
    int boxX = spectW - boxW - 4;
    int boxY = 4;

    p.fillRect(boxX, boxY, boxW, boxH, QColor(0, 0, 0, 180));
    p.setPen(QColor(30, 30, 50));
    p.drawRect(boxX, boxY, boxW, boxH);

    p.setPen(QColor(160, 200, 255));
    for (int i = 0; i < lines.size(); i++) {
        p.drawText(boxX + 4, boxY + 12 + i * lineH, lines[i]);
    }

    // ── Gradient bar (bottom-left of spectrogram) ──
    int barH = 8;
    int barY = spectH - barH - 2;
    int barX = 4;
    int barW2 = 200;
    for (int i = 0; i < barW2; i++) {
        float t = (float)i / barW2;
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

QImage SpectrogramWidget::renderToImage() const
{
    QImage img(size(), QImage::Format_RGB32);
    img.fill(Qt::black);
    const_cast<SpectrogramWidget*>(this)->render(&img);
    return img;
}
