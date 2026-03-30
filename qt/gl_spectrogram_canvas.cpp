#include "gl_spectrogram_canvas.h"

#include "spectrogram_widget.h"

#include <QMatrix4x4>
#include <QPainter>
#include <QVector4D>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>

namespace {
struct Vertex {
    float x;
    float y;
    float u;
    float v;
};

constexpr std::array<Vertex, 4> kQuadVertices{{
    {-1.0f, -1.0f, 0.0f, 1.0f},
    { 1.0f, -1.0f, 1.0f, 1.0f},
    {-1.0f,  1.0f, 0.0f, 0.0f},
    { 1.0f,  1.0f, 1.0f, 0.0f},
}};
}

GlSpectrogramCanvas::GlSpectrogramCanvas(SpectrogramWidget* owner)
    : QOpenGLWidget(owner), owner_(owner), quadBuffer_(QOpenGLBuffer::VertexBuffer)
{
}

GlSpectrogramCanvas::~GlSpectrogramCanvas()
{
    makeCurrent();
    quadBuffer_.destroy();
    delete spectrogramTexture_;
    delete spectrumTexture_;
    delete colorTexture_;
    doneCurrent();
}

void GlSpectrogramCanvas::requestRepaint()
{
    update();
}

void GlSpectrogramCanvas::initializeGL()
{
    initializeOpenGLFunctions();

    quadBuffer_.create();
    quadBuffer_.bind();
    quadBuffer_.allocate(kQuadVertices.data(), static_cast<int>(sizeof(Vertex) * kQuadVertices.size()));
    quadBuffer_.release();

    spectrogramProgram_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
        attribute vec2 position;
        attribute vec2 texCoord;
        varying vec2 vTexCoord;
        uniform mat4 rectTransform;
        void main() {
            vTexCoord = texCoord;
            gl_Position = rectTransform * vec4(position, 0.0, 1.0);
        }
    )");
    spectrogramProgram_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
        varying mediump vec2 vTexCoord;
        uniform sampler2D spectrogramTex;
        void main() {
            gl_FragColor = texture2D(spectrogramTex, vTexCoord);
        }
    )");
    spectrogramProgram_.link();

    histogramProgram_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
        attribute vec2 position;
        attribute vec2 texCoord;
        varying vec2 vTexCoord;
        uniform mat4 rectTransform;
        void main() {
            vTexCoord = texCoord;
            gl_Position = rectTransform * vec4(position, 0.0, 1.0);
        }
    )");
    histogramProgram_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
        varying mediump vec2 vTexCoord;
        uniform sampler2D amplitudeTex;
        uniform sampler2D colorTex;
        uniform vec4 backgroundColor;
        void main() {
            float amp = texture2D(amplitudeTex, vec2(0.5, vTexCoord.y)).r;
            vec4 barColor = texture2D(colorTex, vec2(0.5, vTexCoord.y));
            gl_FragColor = vTexCoord.x <= amp ? barColor : backgroundColor;
        }
    )");
    histogramProgram_.link();

    spectrogramTexture_ = new QOpenGLTexture(QOpenGLTexture::Target2D);
    spectrogramTexture_->create();
    spectrogramTexture_->setWrapMode(QOpenGLTexture::ClampToEdge);
    spectrogramTexture_->setMinificationFilter(QOpenGLTexture::Nearest);
    spectrogramTexture_->setMagnificationFilter(QOpenGLTexture::Nearest);

    spectrumTexture_ = new QOpenGLTexture(QOpenGLTexture::Target2D);
    spectrumTexture_->create();
    spectrumTexture_->setWrapMode(QOpenGLTexture::ClampToEdge);
    spectrumTexture_->setMinificationFilter(QOpenGLTexture::Nearest);
    spectrumTexture_->setMagnificationFilter(QOpenGLTexture::Nearest);

    colorTexture_ = new QOpenGLTexture(QOpenGLTexture::Target2D);
    colorTexture_->create();
    colorTexture_->setWrapMode(QOpenGLTexture::ClampToEdge);
    colorTexture_->setMinificationFilter(QOpenGLTexture::Nearest);
    colorTexture_->setMagnificationFilter(QOpenGLTexture::Nearest);
}

void GlSpectrogramCanvas::paintGL()
{
    QRect spectRect = owner_->spectrogramRect(size());
    QRect histRect = owner_->histogramRect(size());

    glViewport(0, 0, width(), height());
    glDisable(GL_DEPTH_TEST);
    glClearColor(10.0f / 255.0f, 10.0f / 255.0f, 16.0f / 255.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    drawSpectrogram(spectRect);
    drawHistogram(histRect);

    QPainter p(this);
    owner_->paintDecorations(p, size());
}

void GlSpectrogramCanvas::wheelEvent(QWheelEvent* event)
{
    owner_->handleWheelZoom(event->position().toPoint(), event->angleDelta().y(), size());
    event->accept();
}

void GlSpectrogramCanvas::bindQuad(QOpenGLShaderProgram& program)
{
    quadBuffer_.bind();
    int posLoc = program.attributeLocation("position");
    int uvLoc = program.attributeLocation("texCoord");
    program.enableAttributeArray(posLoc);
    program.setAttributeBuffer(posLoc, GL_FLOAT, offsetof(Vertex, x), 2, sizeof(Vertex));
    program.enableAttributeArray(uvLoc);
    program.setAttributeBuffer(uvLoc, GL_FLOAT, offsetof(Vertex, u), 2, sizeof(Vertex));
}

void GlSpectrogramCanvas::releaseQuad(QOpenGLShaderProgram& program)
{
    program.disableAttributeArray("position");
    program.disableAttributeArray("texCoord");
    quadBuffer_.release();
}

QMatrix4x4 GlSpectrogramCanvas::rectTransform(const QRect& rect)
{
    float left = (2.0f * rect.left()) / width() - 1.0f;
    float right = (2.0f * (rect.right() + 1)) / width() - 1.0f;
    float top = 1.0f - (2.0f * rect.top()) / height();
    float bottom = 1.0f - (2.0f * (rect.bottom() + 1)) / height();

    QMatrix4x4 m;
    m.setColumn(0, QVector4D((right - left) / 2.0f, 0, 0, 0));
    m.setColumn(1, QVector4D(0, (top - bottom) / 2.0f, 0, 0));
    m.setColumn(2, QVector4D(0, 0, 1, 0));
    m.setColumn(3, QVector4D((left + right) / 2.0f, (top + bottom) / 2.0f, 0, 1));
    return m;
}

void GlSpectrogramCanvas::ensureTextureStorage(QOpenGLTexture* texture, int widthPx, int heightPx)
{
    if (!texture || widthPx <= 0 || heightPx <= 0) return;
    if (texture->width() == widthPx && texture->height() == heightPx && texture->isStorageAllocated()) return;
    if (texture->isCreated()) texture->destroy();
    texture->create();
    texture->setFormat(QOpenGLTexture::RGBA8_UNorm);
    texture->setSize(widthPx, heightPx);
    texture->allocateStorage(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8);
    texture->setWrapMode(QOpenGLTexture::ClampToEdge);
    texture->setMinificationFilter(QOpenGLTexture::Nearest);
    texture->setMagnificationFilter(QOpenGLTexture::Nearest);
}

void GlSpectrogramCanvas::drawSpectrogram(const QRect& rect)
{
    QImage upload = owner_->spectrogramImage().convertToFormat(QImage::Format_RGBA8888);
    ensureTextureStorage(spectrogramTexture_, upload.width(), upload.height());
    spectrogramTexture_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, upload.constBits());

    spectrogramProgram_.bind();
    spectrogramProgram_.setUniformValue("rectTransform", rectTransform(rect));
    spectrogramProgram_.setUniformValue("spectrogramTex", 0);
    spectrogramTexture_->bind(0);
    bindQuad(spectrogramProgram_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    releaseQuad(spectrogramProgram_);
    spectrogramTexture_->release();
    spectrogramProgram_.release();
}

void GlSpectrogramCanvas::drawHistogram(const QRect& rect)
{
    int nb = owner_->transform_->numBins();
    if (nb < 1) return;

    std::vector<unsigned char> amplitudeRgba(nb * 4, 0);
    std::vector<unsigned char> colorRgba(nb * 4, 0);

    float logMax = std::log(1.0f + owner_->maxAmplitude_ * owner_->gain_);
    for (int fi = 0; fi < nb; fi++) {
        int row = (nb - 1 - fi) * 4;
        float normalized = 0.0f;
        if (fi < static_cast<int>(owner_->spectrum_.size()) && logMax > 0.0f) {
            normalized = std::log(1.0f + owner_->spectrum_[fi] * owner_->gain_) / logMax;
            normalized = std::clamp(normalized, 0.0f, 1.0f);
        }
        amplitudeRgba[row + 0] = static_cast<unsigned char>(normalized * 255.0f);
        amplitudeRgba[row + 1] = amplitudeRgba[row + 0];
        amplitudeRgba[row + 2] = amplitudeRgba[row + 0];
        amplitudeRgba[row + 3] = 255;

        auto [r, g, b] = owner_->colormap(fi < static_cast<int>(owner_->spectrum_.size()) ? owner_->spectrum_[fi] : 0.0f);
        colorRgba[row + 0] = r;
        colorRgba[row + 1] = g;
        colorRgba[row + 2] = b;
        colorRgba[row + 3] = 255;
    }

    ensureTextureStorage(spectrumTexture_, 1, nb);
    ensureTextureStorage(colorTexture_, 1, nb);
    spectrumTexture_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, amplitudeRgba.data());
    colorTexture_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, colorRgba.data());

    histogramProgram_.bind();
    histogramProgram_.setUniformValue("rectTransform", rectTransform(rect));
    histogramProgram_.setUniformValue("amplitudeTex", 0);
    histogramProgram_.setUniformValue("colorTex", 1);
    histogramProgram_.setUniformValue("backgroundColor", QVector4D(12.0f / 255.0f, 12.0f / 255.0f, 20.0f / 255.0f, 1.0f));
    spectrumTexture_->bind(0);
    colorTexture_->bind(1);
    bindQuad(histogramProgram_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    releaseQuad(histogramProgram_);
    colorTexture_->release();
    spectrumTexture_->release();
    histogramProgram_.release();
}
