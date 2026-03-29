#include "spectrogram_widget.h"
#include <QPainter>
#include <QPaintEvent>
#include <cmath>

SpectrogramWidget::SpectrogramWidget(LoiaconoRolling* transform, QWidget* parent)
    : QWidget(parent), transform_(transform)
{
    setMinimumSize(800, 400);
    image_ = QImage(800, 1, QImage::Format_RGB32);
    image_.fill(Qt::black);

    connect(&timer_, &QTimer::timeout, this, &SpectrogramWidget::tick);
    timer_.start(16); // ~60 fps
}

void SpectrogramWidget::resizeEvent(QResizeEvent*)
{
    // Rebuild the image buffer at the new height (= numBins)
    int h = transform_->numBins();
    if (h < 1) h = 1;
    if (image_.height() != h || image_.width() != width()) {
        QImage newImage(width(), h, QImage::Format_RGB32);
        newImage.fill(Qt::black);
        image_ = newImage;
    }
}

void SpectrogramWidget::tick()
{
    int numBins = transform_->numBins();
    if (numBins < 1) return;

    // Ensure image dimensions match
    if (image_.height() != numBins || image_.width() != width()) {
        QImage newImage(width(), numBins, QImage::Format_RGB32);
        newImage.fill(Qt::black);
        image_ = newImage;
    }

    // Get current spectrum from the rolling transform
    transform_->getSpectrum(spectrum_);

    // Auto-adjust max amplitude
    float currentMax = 0;
    for (float v : spectrum_) {
        if (v > currentMax) currentMax = v;
    }
    if (currentMax > maxAmplitude_) {
        maxAmplitude_ = currentMax;
    } else {
        maxAmplitude_ = maxAmplitude_ * 0.998f + currentMax * 0.002f;
    }
    if (maxAmplitude_ < 0.1f) maxAmplitude_ = 0.1f;

    float logMax = std::log(1.0f + maxAmplitude_);

    // Scroll image left by 1 pixel
    int w = image_.width();
    int h = image_.height();
    for (int y = 0; y < h; y++) {
        auto* line = reinterpret_cast<QRgb*>(image_.scanLine(y));
        std::memmove(line, line + 1, (w - 1) * sizeof(QRgb));
    }

    // Write new rightmost column — low freq at bottom (row h-1), high freq at top (row 0)
    for (int fi = 0; fi < numBins && fi < h; fi++) {
        int row = h - 1 - fi;
        float normalized = std::log(1.0f + spectrum_[fi]) / logMax;
        auto [r, g, b] = hotColormap(normalized);
        auto* line = reinterpret_cast<QRgb*>(image_.scanLine(row));
        line[w - 1] = qRgb(r, g, b);
    }

    update();
}

void SpectrogramWidget::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);
    // Scale the internal image to fill the widget
    p.drawImage(rect(), image_);

    // Draw frequency labels
    p.setPen(QColor(120, 120, 160));
    QFont font = p.font();
    font.setPixelSize(10);
    p.setFont(font);

    int numBins = transform_->numBins();
    if (numBins < 2) return;

    double fMin = transform_->binFreqHz(0);
    double fMax = transform_->binFreqHz(numBins - 1);
    double logMin = std::log(fMin);
    double logRange = std::log(fMax) - logMin;

    // Standard label frequencies
    const double labelFreqs[] = {50,100,200,300,440,500,1000,2000,3000,5000,8000,10000,12000};
    for (double f : labelFreqs) {
        if (f < fMin || f > fMax) continue;
        double logPos = (std::log(f) - logMin) / logRange;
        int y = height() - static_cast<int>(logPos * height());
        QString label = f >= 1000 ? QString("%1k").arg(f / 1000.0, 0, 'g', 3)
                                  : QString::number(static_cast<int>(f));
        p.drawText(4, y - 2, label);
        p.drawLine(0, y, 3, y);
    }

    // Time arrow
    p.setPen(QColor(80, 80, 100));
    p.drawText(width() / 2 - 20, height() - 4, "time →");
}

SpectrogramWidget::RGB SpectrogramWidget::hotColormap(float t)
{
    t = std::clamp(t, 0.0f, 1.0f);
    t = std::pow(t, 0.6f); // gamma boost

    uint8_t r, g, b;
    if (t < 0.05f) {
        r = 0; g = 0; b = 0;
    } else if (t < 0.2f) {
        float s = (t - 0.05f) / 0.15f;
        r = 0; g = 0; b = static_cast<uint8_t>(s * 200);
    } else if (t < 0.4f) {
        float s = (t - 0.2f) / 0.2f;
        r = 0; g = static_cast<uint8_t>(s * 255); b = static_cast<uint8_t>(200 + s * 55);
    } else if (t < 0.6f) {
        float s = (t - 0.4f) / 0.2f;
        r = 0; g = 255; b = static_cast<uint8_t>(255 * (1 - s));
    } else if (t < 0.8f) {
        float s = (t - 0.6f) / 0.2f;
        r = static_cast<uint8_t>(s * 255); g = 255; b = 0;
    } else {
        float s = (t - 0.8f) / 0.2f;
        r = 255; g = static_cast<uint8_t>(255 * (1 - s)); b = 0;
    }
    return {r, g, b};
}
