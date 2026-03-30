#pragma once

#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLWidget>
#include <QRect>

class SpectrogramWidget;

class GlSpectrogramCanvas : public QOpenGLWidget, protected QOpenGLFunctions {
    Q_OBJECT
public:
    explicit GlSpectrogramCanvas(SpectrogramWidget* owner);
    ~GlSpectrogramCanvas() override;

    void requestRepaint();

protected:
    void initializeGL() override;
    void paintGL() override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void bindQuad(QOpenGLShaderProgram& program);
    void releaseQuad(QOpenGLShaderProgram& program);
    QMatrix4x4 rectTransform(const QRect& rect);
    void ensureTextureStorage(QOpenGLTexture* texture, int widthPx, int heightPx);
    void drawSpectrogram(const QRect& rect);
    void drawHistogram(const QRect& rect);

    SpectrogramWidget* owner_;
    QOpenGLShaderProgram spectrogramProgram_;
    QOpenGLShaderProgram histogramProgram_;
    QOpenGLBuffer quadBuffer_;
    QOpenGLTexture* spectrogramTexture_ = nullptr;
    QOpenGLTexture* spectrumTexture_ = nullptr;
    QOpenGLTexture* colorTexture_ = nullptr;
};
