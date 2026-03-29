#pragma once
#include <QWidget>
#include <QImage>
#include <QTimer>
#include <vector>
#include "loiacono_rolling.h"

// Horizontal scrolling spectrogram: time on X axis, frequency on Y axis (log scale)
// Reads spectrum from LoiaconoRolling and paints one column per timer tick

class SpectrogramWidget : public QWidget {
    Q_OBJECT
public:
    explicit SpectrogramWidget(LoiaconoRolling* transform, QWidget* parent = nullptr);

    void setMaxAmplitude(float a) { maxAmplitude_ = a; }
    const QImage& spectrogramImage() const { return image_; }
    QImage renderToImage() const;

protected:
    void paintEvent(QPaintEvent*) override;
    void resizeEvent(QResizeEvent*) override;

private slots:
    void tick();

private:
    struct RGB { uint8_t r, g, b; };
    static RGB hotColormap(float t);

    LoiaconoRolling* transform_;
    QImage image_;
    QTimer timer_;
    float maxAmplitude_ = 1.0f;
    std::vector<float> spectrum_;
};
