#pragma once

#include <QMatrix4x4>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>
#include <QOpenGLFunctions>
#include <QOpenGLShaderProgram>
#include <QOpenGLTexture>
#include <QOpenGLWidget>
#include <QRect>

#include "loiacono_rolling.h"

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
    void resizeEvent(QResizeEvent* event) override;
    void wheelEvent(QWheelEvent* event) override;

private:
    void bindQuad(QOpenGLShaderProgram& program);
    void releaseQuad(QOpenGLShaderProgram& program);
    QMatrix4x4 rectTransform(const QRect& rect);
    void ensureTextureStorage(QOpenGLTexture* texture, int widthPx, int heightPx);
    void paintLegacyPath(const QRect& spectRect, const QRect& histRect);
    void paintDirectGpuPath(const QRect& spectRect, const QRect& histRect);
    void drawLegacySpectrogram(const QRect& rect);
    void drawHistogram(const QRect& rect, QOpenGLTexture* amplitudeTexture, QOpenGLTexture* colorTexture);
    bool ensureDirectGpuResources(const QRect& spectRect, const LoiaconoRolling::GpuInputSnapshot& snapshot);
    bool bootstrapDirectRollingState(const LoiaconoRolling::GpuInputSnapshot& snapshot);
    bool runDirectRollingUpdates(const LoiaconoRolling::GpuChunkBatch& batch);
    bool runDirectMagnitudeCompute();
    bool updateDirectTextures(int columnsToAdvance, const QRect& spectRect);
    bool updateDirectStatsFromGpu();

    SpectrogramWidget* owner_;
    QOpenGLShaderProgram spectrogramProgram_;
    QOpenGLShaderProgram histogramProgram_;
    QOpenGLShaderProgram shiftProgram_;
    QOpenGLShaderProgram directBootstrapProgram_;
    QOpenGLShaderProgram directRollingUpdateProgram_;
    QOpenGLShaderProgram directMagnitudeProgram_;
    QOpenGLShaderProgram directTextureProgram_;
    QOpenGLShaderProgram directStatsProgram_;
    QOpenGLBuffer quadBuffer_;
    QOpenGLTexture* spectrogramTexture_ = nullptr;
    QOpenGLTexture* spectrumTexture_ = nullptr;
    QOpenGLTexture* colorTexture_ = nullptr;
    QOpenGLTexture* directSpectrogramFront_ = nullptr;
    QOpenGLTexture* directSpectrogramBack_ = nullptr;
    QOpenGLTexture* directColumnTexture_ = nullptr;
    QOpenGLTexture* directAmplitudeTexture_ = nullptr;
    QOpenGLTexture* directColorTexture_ = nullptr;
    QOpenGLFramebufferObject* directSpectrogramFbo_ = nullptr;
    GLuint directBuffers_[11] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool directBuffersInitialized_ = false;
    int directNumBins_ = 0;
    int directSignalLength_ = 0;
    int directHistoryRevision_ = -1;
    bool directRollingBootstrapped_ = false;
};
