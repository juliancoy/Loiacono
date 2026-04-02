#include "gl_spectrogram_canvas.h"

#include "spectrogram_widget.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QImage>
#include <QMatrix4x4>
#include <QMouseEvent>
#include <QOpenGLExtraFunctions>
#include <QPainter>
#include <QResizeEvent>
#include <QVector4D>
#include <QWheelEvent>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

namespace {
struct Vertex {
    float x;
    float y;
    float u;
    float v;
};

constexpr int THREADS_PER_WORKGROUP = 128;
constexpr int MAX_DIRECT_GPU_CHUNK = 2048;
constexpr int DIRECT_BUF_NEW_CHUNK = 0;
constexpr int DIRECT_BUF_OLD_CHUNK = 1;
constexpr int DIRECT_BUF_FREQ = 2;
constexpr int DIRECT_BUF_NORM = 3;
constexpr int DIRECT_BUF_WINDOW = 4;
constexpr int DIRECT_BUF_TR = 5;
constexpr int DIRECT_BUF_TI = 6;
constexpr int DIRECT_BUF_SPECTRUM = 7;
constexpr int DIRECT_BUF_STATS = 8;
constexpr int DIRECT_BUF_RING = 9;
constexpr int DIRECT_BUF_OFFSET = 10;

constexpr std::array<Vertex, 4> kQuadVertices{{
    {-1.0f, -1.0f, 0.0f, 1.0f},
    { 1.0f, -1.0f, 1.0f, 1.0f},
    {-1.0f,  1.0f, 0.0f, 0.0f},
    { 1.0f,  1.0f, 1.0f, 0.0f},
}};

void ensureStorage(QOpenGLTexture*& texture, int widthPx, int heightPx)
{
    if (!texture) {
        texture = new QOpenGLTexture(QOpenGLTexture::Target2D);
        texture->create();
    }
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
}

GlSpectrogramCanvas::GlSpectrogramCanvas(SpectrogramWidget* owner)
    : QOpenGLWidget(owner), owner_(owner), quadBuffer_(QOpenGLBuffer::VertexBuffer)
{
    setMouseTracking(true);
}

GlSpectrogramCanvas::~GlSpectrogramCanvas()
{
    makeCurrent();
    quadBuffer_.destroy();
    delete spectrogramTexture_;
    delete spectrumTexture_;
    delete colorTexture_;
    delete directSpectrogramFront_;
    delete directSpectrogramBack_;
    delete directColumnTexture_;
    delete directAmplitudeTexture_;
    delete directColorTexture_;
    delete directSpectrogramFbo_;
    if (directBuffers_[0]) {
        glDeleteBuffers(11, directBuffers_);
    }
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

    shiftProgram_.addShaderFromSourceCode(QOpenGLShader::Vertex, R"(
        attribute vec2 position;
        attribute vec2 texCoord;
        varying vec2 vTexCoord;
        void main() {
            vTexCoord = texCoord;
            gl_Position = vec4(position, 0.0, 1.0);
        }
    )");
    shiftProgram_.addShaderFromSourceCode(QOpenGLShader::Fragment, R"(
        varying mediump vec2 vTexCoord;
        uniform sampler2D prevTex;
        uniform sampler2D columnTex;
        uniform float shiftFraction;
        uniform float columnFraction;
        void main() {
            if (vTexCoord.x >= 1.0 - columnFraction) {
                float localX = (vTexCoord.x - (1.0 - columnFraction)) / max(columnFraction, 0.0001);
                gl_FragColor = texture2D(columnTex, vec2(localX, vTexCoord.y));
            } else {
                gl_FragColor = texture2D(prevTex, vec2(vTexCoord.x + shiftFraction, vTexCoord.y));
            }
        }
    )");
    shiftProgram_.link();

    directBootstrapProgram_.addShaderFromSourceCode(QOpenGLShader::Compute, R"(
        #version 430
        layout(local_size_x = 128) in;
        layout(std430, binding = 0) readonly buffer RingBuf { float ringData[]; };
        layout(std430, binding = 1) readonly buffer FreqBuf { float freqs[]; };
        layout(std430, binding = 2) readonly buffer NormBuf { float norms[]; };
        layout(std430, binding = 3) readonly buffer WindowBuf { int windowLens[]; };
        layout(std430, binding = 4) readonly buffer OffsetBuf { uint offsetData[16]; };
        layout(std430, binding = 5) buffer TrBuf { float trState[]; };
        layout(std430, binding = 6) buffer TiBuf { float tiState[]; };
        uniform int signalLength;
        uniform int numBins;
        uniform float sampleCountEnd;
        shared float sharedTr[128];
        shared float sharedTi[128];
        void main() {
            uint bin = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (bin >= uint(numBins)) return;
            float freq = freqs[bin];
            float norm = norms[bin];
            int windowLen = windowLens[bin];
            uint offset = offsetData[0];
            float endSampleCount = sampleCountEnd;
            float tr = 0.0;
            float ti = 0.0;
            for (int k = int(tid); k < windowLen; k += 128) {
                int readIndex = (int(offset) - windowLen + k + signalLength) % signalLength;
                float sampleIx = endSampleCount - float(windowLen) + float(k);
                float angle = 6.283185307179586 * freq * sampleIx;
                float sampleValue = ringData[readIndex];
                tr += sampleValue * cos(angle) * norm;
                ti -= sampleValue * sin(angle) * norm;
            }
            sharedTr[tid] = tr;
            sharedTi[tid] = ti;
            barrier();
            for (uint stride = 64u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    sharedTr[tid] += sharedTr[tid + stride];
                    sharedTi[tid] += sharedTi[tid + stride];
                }
                barrier();
            }
            if (tid == 0u) {
                trState[bin] = sharedTr[0];
                tiState[bin] = sharedTi[0];
            }
        }
    )");
    directBootstrapProgram_.link();

    directRollingUpdateProgram_.addShaderFromSourceCode(QOpenGLShader::Compute, R"(
        #version 430
        layout(local_size_x = 128) in;
        layout(std430, binding = 0) readonly buffer NewChunkBuf { float newChunk[]; };
        layout(std430, binding = 1) readonly buffer RingBuf { float ringData[]; };
        layout(std430, binding = 2) readonly buffer FreqBuf { float freqs[]; };
        layout(std430, binding = 3) readonly buffer NormBuf { float norms[]; };
        layout(std430, binding = 4) readonly buffer WindowBuf { int windowLens[]; };
        layout(std430, binding = 5) buffer TrBuf { float trState[]; };
        layout(std430, binding = 6) buffer TiBuf { float tiState[]; };
        uniform int chunkLength;
        uniform int signalLength;
        uniform int numBins;
        uniform uint sampleBase;
        uniform int ringHeadStart;
        uniform float leakiness;
        shared float sharedTr[128];
        shared float sharedTi[128];
        void main() {
            uint bin = gl_WorkGroupID.x;
            uint tid = gl_LocalInvocationID.x;
            if (bin >= uint(numBins)) return;
            float freq = freqs[bin];
            float norm = norms[bin];
            int windowLen = windowLens[bin];
            float chunkLeak = pow(leakiness, float(chunkLength));
            float tr = 0.0;
            float ti = 0.0;
            for (int i = int(tid); i < chunkLength; i += 128) {
                uint sampleIx = sampleBase + uint(i);
                float angle = 6.283185307179586 * freq * sampleIx;
                float sampleValue = newChunk[i];
                tr += sampleValue * cos(angle) * norm;
                ti -= sampleValue * sin(angle) * norm;
                if (sampleIx >= uint(windowLen)) {
                    int oldIdx = (ringHeadStart + i - windowLen + signalLength) % signalLength;
                    float oldAngle = 6.283185307179586 * freq * float(sampleIx - uint(windowLen));
                    float oldSampleValue = ringData[oldIdx];
                    tr -= oldSampleValue * cos(oldAngle) * norm;
                    ti += oldSampleValue * sin(oldAngle) * norm;
                }
            }
            sharedTr[tid] = tr;
            sharedTi[tid] = ti;
            barrier();
            for (uint stride = 64u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    sharedTr[tid] += sharedTr[tid + stride];
                    sharedTi[tid] += sharedTi[tid + stride];
                }
                barrier();
            }
            if (tid == 0u) {
                trState[bin] = trState[bin] * chunkLeak + sharedTr[0];
                tiState[bin] = tiState[bin] * chunkLeak + sharedTi[0];
            }
        }
    )");
    directRollingUpdateProgram_.link();

    directMagnitudeProgram_.addShaderFromSourceCode(QOpenGLShader::Compute, R"(
        #version 430
        layout(local_size_x = 128) in;
        layout(std430, binding = 0) readonly buffer TrBuf { float trState[]; };
        layout(std430, binding = 1) readonly buffer TiBuf { float tiState[]; };
        layout(std430, binding = 2) writeonly buffer SpectrumBuf { float spectrum[]; };
        uniform int numBins;
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= uint(numBins)) return;
            float tr = trState[idx];
            float ti = tiState[idx];
            spectrum[idx] = sqrt(tr * tr + ti * ti);
        }
    )");
    directMagnitudeProgram_.link();

    directTextureProgram_.addShaderFromSourceCode(QOpenGLShader::Compute, R"(
        #version 430
        layout(local_size_x = 128) in;
        layout(std430, binding = 0) readonly buffer SpectrumBuf { float spectrum[]; };
        layout(std430, binding = 1) readonly buffer StatsBuf { vec4 stats; };
        layout(rgba8, binding = 0) uniform image2D columnImg;
        layout(rgba8, binding = 1) uniform image2D amplitudeImg;
        layout(rgba8, binding = 2) uniform image2D colorImg;
        uniform float gain;
        uniform float gammaValue;
        uniform float floorValue;
        uniform int numBins;

        vec4 mapColor(float amplitude) {
            float maxAmplitude = max(stats.z, 0.01);
            float logMax = log(1.0 + maxAmplitude * gain);
            float t = logMax > 0.0 ? log(1.0 + amplitude * gain) / logMax : 0.0;
            t = clamp(t, 0.0, 1.0);
            if (t < floorValue) return vec4(0.0, 0.0, 0.0, 1.0);
            t = (t - floorValue) / (1.0 - floorValue);
            t = pow(t, gammaValue);
            if (t < 0.15) {
                float s = t / 0.15;
                return vec4(0.0, 0.0, s * 200.0 / 255.0, 1.0);
            } else if (t < 0.35) {
                float s = (t - 0.15) / 0.2;
                return vec4(0.0, s, (200.0 + s * 55.0) / 255.0, 1.0);
            } else if (t < 0.55) {
                float s = (t - 0.35) / 0.2;
                return vec4(0.0, 1.0, 1.0 - s, 1.0);
            } else if (t < 0.75) {
                float s = (t - 0.55) / 0.2;
                return vec4(s, 1.0, 0.0, 1.0);
            } else {
                float s = (t - 0.75) / 0.25;
                return vec4(1.0, 1.0 - s * 0.6, s * 180.0 / 255.0, 1.0);
            }
        }

        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= uint(numBins)) return;
            float amp = spectrum[idx];
            float maxAmplitude = max(stats.z, 0.01);
            float logMax = log(1.0 + maxAmplitude * gain);
            float normalized = logMax > 0.0 ? log(1.0 + amp * gain) / logMax : 0.0;
            normalized = clamp(normalized, 0.0, 1.0);
            vec4 color = mapColor(amp);
            int y = numBins - 1 - int(idx);
            imageStore(columnImg, ivec2(0, y), color);
            imageStore(amplitudeImg, ivec2(0, y), vec4(normalized, normalized, normalized, 1.0));
            imageStore(colorImg, ivec2(0, y), color);
        }
    )");
    directTextureProgram_.link();

    directStatsProgram_.addShaderFromSourceCode(QOpenGLShader::Compute, R"(
        #version 430
        layout(local_size_x = 128) in;
        layout(std430, binding = 0) readonly buffer SpectrumBuf { float spectrum[]; };
        layout(std430, binding = 1) buffer StatsBuf { vec4 stats; };
        uniform int numBins;
        shared float sharedAmp[128];
        shared uint sharedIdx[128];
        void main() {
            uint tid = gl_LocalInvocationID.x;
            float bestAmp = -1.0;
            uint bestIdx = 0u;
            for (uint idx = tid; idx < uint(numBins); idx += gl_WorkGroupSize.x) {
                float amp = spectrum[idx];
                if (amp > bestAmp) {
                    bestAmp = amp;
                    bestIdx = idx;
                }
            }
            sharedAmp[tid] = bestAmp;
            sharedIdx[tid] = bestIdx;
            barrier();
            for (uint stride = 64u; stride > 0u; stride >>= 1u) {
                if (tid < stride && sharedAmp[tid + stride] > sharedAmp[tid]) {
                    sharedAmp[tid] = sharedAmp[tid + stride];
                    sharedIdx[tid] = sharedIdx[tid + stride];
                }
                barrier();
            }
            if (tid == 0u) {
                float currentMax = max(sharedAmp[0], 0.0);
                float displayMax = max(max(stats.z * 0.997 + currentMax * 0.003, currentMax), 0.01);
                stats = vec4(currentMax, float(sharedIdx[0]), displayMax, 0.0);
            }
        }
    )");
    directStatsProgram_.link();

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
    // Dark background
    glClearColor(10.0f / 255.0f, 10.0f / 255.0f, 16.0f / 255.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Ensure image is resized before painting
    QImage img = owner_->spectrogramImage();
    if (img.width() != spectRect.width() || img.height() != spectRect.height()) {
        owner_->onCanvasResized();
        spectRect = owner_->spectrogramRect(size());
        histRect = owner_->histogramRect(size());
    }

    if (owner_->useDirectGpuPipeline()) {
        paintDirectGpuPath(spectRect, histRect);
    } else {
        paintLegacyPath(spectRect, histRect);
    }

    QPainter p(this);
    owner_->paintDecorations(p, size());
}

void GlSpectrogramCanvas::resizeEvent(QResizeEvent* event)
{
    QOpenGLWidget::resizeEvent(event);
    // Don't call update() here - parent will handle it after resizing the image
    // Just notify the parent that resize happened
    owner_->onCanvasResized();
}

void GlSpectrogramCanvas::wheelEvent(QWheelEvent* event)
{
    owner_->handleWheelZoom(event->position().toPoint(), event->angleDelta().y(), size());
    event->accept();
}

void GlSpectrogramCanvas::mouseMoveEvent(QMouseEvent* event)
{
    owner_->updateHoverCursor(event->position().toPoint(), size());
    QOpenGLWidget::mouseMoveEvent(event);
}

void GlSpectrogramCanvas::leaveEvent(QEvent* event)
{
    unsetCursor();
    QOpenGLWidget::leaveEvent(event);
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

void GlSpectrogramCanvas::paintLegacyPath(const QRect& spectRect, const QRect& histRect)
{
    drawLegacySpectrogram(spectRect);
    drawHistogram(histRect, spectrumTexture_, colorTexture_);
}

void GlSpectrogramCanvas::paintDirectGpuPath(const QRect& spectRect, const QRect& histRect)
{
    auto snapshot = owner_->transform_->gpuInputSnapshot();
    auto batch = owner_->transform_->takePendingGpuChunks();
    if (!ensureDirectGpuResources(spectRect, snapshot)) {
        paintLegacyPath(spectRect, histRect);
        return;
    }

    bool needBootstrap = !directRollingBootstrapped_ || batch.overflowed;
    if (needBootstrap) {
        if (!bootstrapDirectRollingState(snapshot)) {
            paintLegacyPath(spectRect, histRect);
            return;
        }
    } else if (!runDirectRollingUpdates(batch)) {
        paintLegacyPath(spectRect, histRect);
        return;
    }

    if (!runDirectMagnitudeCompute()) {
        paintLegacyPath(spectRect, histRect);
        return;
    }

    if (!updateDirectStatsFromGpu()) {
        paintLegacyPath(spectRect, histRect);
        return;
    }
    if (!updateDirectTextures(owner_->pendingGpuColumns_, spectRect)) {
        paintLegacyPath(spectRect, histRect);
        return;
    }
    owner_->pendingGpuColumns_ = 0;

    spectrogramProgram_.bind();
    spectrogramProgram_.setUniformValue("rectTransform", rectTransform(spectRect));
    spectrogramProgram_.setUniformValue("spectrogramTex", 0);
    directSpectrogramFront_->bind(0);
    bindQuad(spectrogramProgram_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    releaseQuad(spectrogramProgram_);
    directSpectrogramFront_->release();
    spectrogramProgram_.release();

    drawHistogram(histRect, directAmplitudeTexture_, directColorTexture_);
}

void GlSpectrogramCanvas::drawLegacySpectrogram(const QRect& rect)
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

void GlSpectrogramCanvas::drawHistogram(const QRect& rect, QOpenGLTexture* amplitudeTexture, QOpenGLTexture* colorTexture)
{
    int nb = owner_->transform_->numBins();
    if (nb < 1) return;

    if (amplitudeTexture == spectrumTexture_ && colorTexture == colorTexture_) {
        std::vector<unsigned char> amplitudeRgba(nb * 4, 0);
        std::vector<unsigned char> colorRgba(nb * 4, 0);

        for (int fi = 0; fi < nb; fi++) {
            int row = (nb - 1 - fi) * 4;
            float normalized = fi < static_cast<int>(owner_->spectrum_.size())
                ? owner_->visualLevel(owner_->spectrum_[fi])
                : 0.0f;
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
    }

    histogramProgram_.bind();
    histogramProgram_.setUniformValue("rectTransform", rectTransform(rect));
    histogramProgram_.setUniformValue("amplitudeTex", 0);
    histogramProgram_.setUniformValue("colorTex", 1);
    histogramProgram_.setUniformValue("backgroundColor", QVector4D(20.0f / 255.0f, 20.0f / 255.0f, 35.0f / 255.0f, 1.0f));  // Dark blue-gray background
    amplitudeTexture->bind(0);
    colorTexture->bind(1);
    bindQuad(histogramProgram_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    releaseQuad(histogramProgram_);
    colorTexture->release();
    amplitudeTexture->release();
    histogramProgram_.release();
}

bool GlSpectrogramCanvas::ensureDirectGpuResources(const QRect& spectRect, const LoiaconoRolling::GpuInputSnapshot& snapshot)
{
    const int numBins = snapshot.numBins;
    if (numBins <= 0) return false;
    if (!directBootstrapProgram_.isLinked() ||
        !directRollingUpdateProgram_.isLinked() ||
        !directMagnitudeProgram_.isLinked() ||
        !directTextureProgram_.isLinked() ||
        !directStatsProgram_.isLinked()) {
        return false;
    }

    ensureStorage(directSpectrogramFront_, spectRect.width(), spectRect.height());
    ensureStorage(directSpectrogramBack_, spectRect.width(), spectRect.height());
    ensureStorage(directColumnTexture_, 1, numBins);
    ensureStorage(directAmplitudeTexture_, 1, numBins);
    ensureStorage(directColorTexture_, 1, numBins);

    if (!directSpectrogramFbo_ || directSpectrogramFbo_->size() != spectRect.size()) {
        delete directSpectrogramFbo_;
        QOpenGLFramebufferObjectFormat format;
        format.setAttachment(QOpenGLFramebufferObject::NoAttachment);
        directSpectrogramFbo_ = new QOpenGLFramebufferObject(spectRect.size(), format);
    }

    if (!directBuffersInitialized_) {
        glGenBuffers(11, directBuffers_);
        directBuffersInitialized_ = true;
    }

    const int signalLength = static_cast<int>(snapshot.ring.size());
    if (directSignalLength_ != signalLength || directNumBins_ != numBins) {
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_NEW_CHUNK]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_DIRECT_GPU_CHUNK * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_OLD_CHUNK]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, MAX_DIRECT_GPU_CHUNK * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_TR]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max(1, numBins) * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_TI]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max(1, numBins) * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_SPECTRUM]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max(1, numBins) * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_RING]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, signalLength * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_FREQ]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max(1, numBins) * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_NORM]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max(1, numBins) * static_cast<GLsizeiptr>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_WINDOW]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, std::max(1, numBins) * static_cast<GLsizeiptr>(sizeof(int)), nullptr, GL_DYNAMIC_DRAW);
        unsigned int offsets[16] = {};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_OFFSET]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(offsets), offsets, GL_DYNAMIC_DRAW);
        std::array<float, 4> zeroStats{0.f, 0.f, 0.f, 0.f};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_STATS]);
        glBufferData(GL_SHADER_STORAGE_BUFFER, static_cast<GLsizeiptr>(zeroStats.size() * sizeof(float)), zeroStats.data(), GL_DYNAMIC_DRAW);

        directSignalLength_ = signalLength;
        directNumBins_ = numBins;
        directHistoryRevision_ = -1;
        directRollingBootstrapped_ = false;
    }

    std::vector<float> freqFloats(numBins);
    std::vector<float> normFloats(numBins);
    for (int i = 0; i < numBins; ++i) {
        freqFloats[i] = static_cast<float>(snapshot.freqs[i]);
        normFloats[i] = static_cast<float>(snapshot.norms[i]);
    }
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_FREQ]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numBins * static_cast<GLsizeiptr>(sizeof(float)), freqFloats.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_NORM]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numBins * static_cast<GLsizeiptr>(sizeof(float)), normFloats.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_WINDOW]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numBins * static_cast<GLsizeiptr>(sizeof(int)), snapshot.windowLens.data());

    if (directHistoryRevision_ != owner_->historyRevision_) {
        directSpectrogramFront_->bind();
        std::vector<unsigned char> zeros(static_cast<size_t>(spectRect.width()) * spectRect.height() * 4, 0);
        directSpectrogramFront_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, zeros.data());
        directSpectrogramFront_->release();
        directSpectrogramBack_->bind();
        directSpectrogramBack_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, zeros.data());
        directSpectrogramBack_->release();
        std::array<float, 4> zeroStats{0.f, 0.f, 1.f, 0.f};
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_STATS]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        0,
                        static_cast<GLsizeiptr>(zeroStats.size() * sizeof(float)),
                        zeroStats.data());
        directHistoryRevision_ = owner_->historyRevision_;
        directRollingBootstrapped_ = false;
    }

    return true;
}

bool GlSpectrogramCanvas::bootstrapDirectRollingState(const LoiaconoRolling::GpuInputSnapshot& snapshot)
{
    if (!directBootstrapProgram_.isLinked()) return false;
    auto* f = context()->extraFunctions();
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_RING]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                    0,
                    static_cast<GLsizeiptr>(snapshot.ring.size() * sizeof(float)),
                    snapshot.ring.data());
    unsigned int offsets[16] = {};
    offsets[0] = snapshot.offset;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_OFFSET]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(offsets), offsets);

    std::vector<float> zeros(std::max(1, directNumBins_), 0.0f);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_TR]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, directNumBins_ * static_cast<GLsizeiptr>(sizeof(float)), zeros.data());
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_TI]);
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, directNumBins_ * static_cast<GLsizeiptr>(sizeof(float)), zeros.data());

    directBootstrapProgram_.bind();
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directBuffers_[DIRECT_BUF_RING]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, directBuffers_[DIRECT_BUF_FREQ]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, directBuffers_[DIRECT_BUF_NORM]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, directBuffers_[DIRECT_BUF_WINDOW]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, directBuffers_[DIRECT_BUF_OFFSET]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, directBuffers_[DIRECT_BUF_TR]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, directBuffers_[DIRECT_BUF_TI]);
    directBootstrapProgram_.setUniformValue("signalLength", directSignalLength_);
    directBootstrapProgram_.setUniformValue("numBins", directNumBins_);
    directBootstrapProgram_.setUniformValue("sampleCountEnd", static_cast<float>(snapshot.sampleCount));
    f->glDispatchCompute(static_cast<GLuint>(std::max(1, directNumBins_)), 1, 1);
    f->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    directBootstrapProgram_.release();
    directRollingBootstrapped_ = true;
    return true;
}

bool GlSpectrogramCanvas::runDirectRollingUpdates(const LoiaconoRolling::GpuChunkBatch& batch)
{
    if (!directRollingUpdateProgram_.isLinked()) return false;
    auto* f = context()->extraFunctions();
    directRollingUpdateProgram_.bind();
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, directBuffers_[DIRECT_BUF_FREQ]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, directBuffers_[DIRECT_BUF_NORM]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, directBuffers_[DIRECT_BUF_WINDOW]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, directBuffers_[DIRECT_BUF_TR]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, directBuffers_[DIRECT_BUF_TI]);
    directRollingUpdateProgram_.setUniformValue("numBins", directNumBins_);
    directRollingUpdateProgram_.setUniformValue("leakiness", static_cast<float>(owner_->transform_->leakiness()));
    for (const auto& chunk : batch.chunks) {
        const int count = static_cast<int>(chunk.newSamples.size());
        if (count <= 0) continue;
        if (count > MAX_DIRECT_GPU_CHUNK) {
            directRollingUpdateProgram_.release();
            return false;
        }
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_NEW_CHUNK]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * static_cast<GLsizeiptr>(sizeof(float)), chunk.newSamples.data());
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directBuffers_[DIRECT_BUF_NEW_CHUNK]);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, directBuffers_[DIRECT_BUF_RING]);
        directRollingUpdateProgram_.setUniformValue("chunkLength", count);
        directRollingUpdateProgram_.setUniformValue("signalLength", directSignalLength_);
        directRollingUpdateProgram_.setUniformValue("sampleBase", static_cast<GLuint>(chunk.startSampleCount & 0xffffffffu));
        directRollingUpdateProgram_.setUniformValue("ringHeadStart", chunk.ringHeadStart);
        f->glDispatchCompute(static_cast<GLuint>(std::max(1, directNumBins_)), 1, 1);

        const int firstChunk = std::min(count, directSignalLength_ - chunk.ringHeadStart);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, directBuffers_[DIRECT_BUF_RING]);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                        chunk.ringHeadStart * static_cast<GLsizeiptr>(sizeof(float)),
                        firstChunk * static_cast<GLsizeiptr>(sizeof(float)),
                        chunk.newSamples.data());
        if (firstChunk < count) {
            glBufferSubData(GL_SHADER_STORAGE_BUFFER,
                            0,
                            (count - firstChunk) * static_cast<GLsizeiptr>(sizeof(float)),
                            chunk.newSamples.data() + firstChunk);
        }
    }
    f->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
    directRollingUpdateProgram_.release();
    return true;
}

bool GlSpectrogramCanvas::runDirectMagnitudeCompute()
{
    if (!directMagnitudeProgram_.isLinked()) return false;
    auto* f = context()->extraFunctions();
    directMagnitudeProgram_.bind();
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directBuffers_[DIRECT_BUF_TR]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, directBuffers_[DIRECT_BUF_TI]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, directBuffers_[DIRECT_BUF_SPECTRUM]);
    directMagnitudeProgram_.setUniformValue("numBins", directNumBins_);
    f->glDispatchCompute(static_cast<GLuint>((directNumBins_ + THREADS_PER_WORKGROUP - 1) / THREADS_PER_WORKGROUP), 1, 1);
    f->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    directMagnitudeProgram_.release();
    return true;
}

bool GlSpectrogramCanvas::updateDirectStatsFromGpu()
{
    if (directNumBins_ <= 0) return false;

    auto* f = context()->extraFunctions();
    directStatsProgram_.bind();
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directBuffers_[DIRECT_BUF_SPECTRUM]);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, directBuffers_[DIRECT_BUF_STATS]);
    directStatsProgram_.setUniformValue("numBins", directNumBins_);
    f->glDispatchCompute(1, 1, 1);
    f->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    directStatsProgram_.release();
    return true;
}

bool GlSpectrogramCanvas::updateDirectTextures(int columnsToAdvance, const QRect& spectRect)
{
    if (columnsToAdvance <= 0) return true;
    auto* f = context()->extraFunctions();

    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, directBuffers_[DIRECT_BUF_SPECTRUM]);

    directTextureProgram_.bind();
    directTextureProgram_.setUniformValue("gain", owner_->gain_);
    directTextureProgram_.setUniformValue("gammaValue", owner_->gamma_);
    directTextureProgram_.setUniformValue("floorValue", owner_->floor_);
    directTextureProgram_.setUniformValue("numBins", directNumBins_);

    f->glBindImageTexture(0, directColumnTexture_->textureId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    f->glBindImageTexture(1, directAmplitudeTexture_->textureId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    f->glBindImageTexture(2, directColorTexture_->textureId(), 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
    f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, directBuffers_[DIRECT_BUF_STATS]);
    f->glDispatchCompute(static_cast<GLuint>((directNumBins_ + THREADS_PER_WORKGROUP - 1) / THREADS_PER_WORKGROUP), 1, 1);
    f->glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);
    directTextureProgram_.release();

    if (columnsToAdvance >= spectRect.width()) {
        directSpectrogramFront_->bind();
        std::vector<unsigned char> zeros(static_cast<size_t>(spectRect.width()) * spectRect.height() * 4, 0);
        directSpectrogramFront_->setData(QOpenGLTexture::RGBA, QOpenGLTexture::UInt8, zeros.data());
        directSpectrogramFront_->release();
    }

    glBindFramebuffer(GL_FRAMEBUFFER, directSpectrogramFbo_->handle());
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, directSpectrogramBack_->textureId(), 0);
    glViewport(0, 0, spectRect.width(), spectRect.height());
    glClearColor(0.f, 0.f, 0.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    shiftProgram_.bind();
    shiftProgram_.setUniformValue("prevTex", 0);
    shiftProgram_.setUniformValue("columnTex", 1);
    shiftProgram_.setUniformValue("shiftFraction", static_cast<float>(columnsToAdvance) / std::max(1, spectRect.width()));
    shiftProgram_.setUniformValue("columnFraction", static_cast<float>(columnsToAdvance) / std::max(1, spectRect.width()));
    directSpectrogramFront_->bind(0);
    directColumnTexture_->bind(1);
    bindQuad(shiftProgram_);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    releaseQuad(shiftProgram_);
    directColumnTexture_->release();
    directSpectrogramFront_->release();
    shiftProgram_.release();

    // Detach texture from FBO before swapping to avoid "texture is attached" error
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, 0, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, defaultFramebufferObject());
    std::swap(directSpectrogramFront_, directSpectrogramBack_);
    glViewport(0, 0, width(), height());
    return true;
}
