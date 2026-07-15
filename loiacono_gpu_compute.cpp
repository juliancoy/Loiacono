#include "loiacono_gpu_compute.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QSurfaceFormat>
#include <QStringList>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr int THREADS_PER_WORKGROUP = 128;

QString shaderTemplateName(int algorithmMode)
{
    switch (algorithmMode) {
    case 1:
        return QStringLiteral("fft_generic.comp.template");
    case 2:
        return QStringLiteral("goertzel_generic.comp.template");
    case 0:
    default:
        return QStringLiteral("loiacono_generic.comp.template");
    }
}

QString loadShaderTemplate(int algorithmMode)
{
    const QString shaderFile = shaderTemplateName(algorithmMode);
    const QStringList candidates = {
#ifdef LOIACONO_SHADER_DIR
        QDir(QStringLiteral(LOIACONO_SHADER_DIR)).absoluteFilePath(shaderFile),
#endif
        QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../../shaders/%1").arg(shaderFile)),
        QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("../loiacono/shaders/%1").arg(shaderFile)),
        QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(QStringLiteral("shaders/%1").arg(shaderFile))
    };
    for (const QString &path : candidates) {
        QFile file(path);
        if (file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return QString::fromUtf8(file.readAll());
        }
    }
    return {};
}

QString buildShaderSource(int signalLength, int algorithmMode)
{
    QString source = loadShaderTemplate(algorithmMode);
    if (source.isEmpty()) return {};

    const QString defines = QString(
        "#define PI 3.14159265358979323846\n"
        "#define SIGNAL_LENGTH %1\n"
        "#define THREADS_PER_WORKGROUP %2\n")
        .arg(signalLength)
        .arg(THREADS_PER_WORKGROUP);

    const QString buffers =
        "layout(std430, binding = 0) buffer x_buf { readonly float x[SIGNAL_LENGTH]; };\n"
        "layout(std430, binding = 1) buffer L_buf { writeonly float L[]; };\n"
        "layout(std430, binding = 2) buffer P_buf { writeonly float P[]; };\n"
        "layout(std430, binding = 3) buffer f_buf { readonly float f[]; };\n"
        "layout(std430, binding = 4) buffer norm_buf { readonly float norm[]; };\n"
        "layout(std430, binding = 5) buffer window_buf { readonly int windowLen[]; };\n"
        "layout(std430, binding = 6) buffer params_buf { readonly uint params[16]; };\n";

    source.replace("DEFINE_STRING", defines);
    source.replace("BUFFERS_STRING", buffers);
    source.replace("LEAKINESS_DECL", "uniform float leakiness;\n");
    source.replace("LEAKINESS_VALUE", "leakiness");
    return source;
}
}

class LoiaconoGpuCompute::Impl {
public:
    ~Impl()
    {
        if (!context_ || !surface_) return;
        context_->makeCurrent(surface_.get());
        auto* f = context_->extraFunctions();
        if (program_) f->glDeleteProgram(program_);
        for (auto& fence : outputFences_) {
            if (fence) {
                f->glDeleteSync(fence);
                fence = nullptr;
            }
        }
        if (buffers_[0]) f->glDeleteBuffers(9, buffers_);
        context_->doneCurrent();
    }

    bool available() const { return initialized_; }

    bool configure(int signalLength,
                   int numBins,
                   const std::vector<double>& freqs,
                   const std::vector<double>& norms,
                   const std::vector<int>& windowLens,
                   int algorithmMode,
                   int windowMode,
                   int normalizationMode,
                   int fftLength)
    {
        if (!ensureContext()) return false;
        if (!context_->makeCurrent(surface_.get())) return false;
        auto* f = context_->extraFunctions();

        const bool needsProgramRebuild = signalLength_ != signalLength
            || algorithmMode_ != algorithmMode
            || program_ == 0;
        if (needsProgramRebuild) {
            if (program_) {
                f->glDeleteProgram(program_);
                program_ = 0;
            }

            const QString source = buildShaderSource(signalLength, algorithmMode);
            if (source.isEmpty()) {
                context_->doneCurrent();
                initialized_ = false;
                return false;
            }

            const QByteArray src = source.toUtf8();
            GLuint shader = f->glCreateShader(GL_COMPUTE_SHADER);
            const char* ptr = src.constData();
            f->glShaderSource(shader, 1, &ptr, nullptr);
            f->glCompileShader(shader);

            GLint ok = GL_FALSE;
            f->glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
            if (!ok) {
                f->glDeleteShader(shader);
                context_->doneCurrent();
                initialized_ = false;
                return false;
            }

            program_ = f->glCreateProgram();
            f->glAttachShader(program_, shader);
            f->glLinkProgram(program_);
            f->glDeleteShader(shader);

            f->glGetProgramiv(program_, GL_LINK_STATUS, &ok);
            if (!ok) {
                f->glDeleteProgram(program_);
                program_ = 0;
                context_->doneCurrent();
                initialized_ = false;
                return false;
            }

            signalLength_ = signalLength;
            algorithmMode_ = algorithmMode;
        }

        if (!buffersInitialized_) {
            f->glGenBuffers(9, buffers_);
            buffersInitialized_ = true;
        }

        std::vector<float> freqFloats(numBins);
        std::vector<float> normFloats(numBins);
        for (int i = 0; i < numBins; ++i) {
            freqFloats[static_cast<size_t>(i)] = static_cast<float>(freqs[static_cast<size_t>(i)]);
            normFloats[static_cast<size_t>(i)] = static_cast<float>(norms[static_cast<size_t>(i)]);
        }

        bindBufferData(f, buffers_[0], signalLength * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[1], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[2], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[7], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[8], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[3], std::max(1, numBins) * static_cast<int>(sizeof(float)), freqFloats.data(), GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[4], std::max(1, numBins) * static_cast<int>(sizeof(float)), normFloats.data(), GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[5], std::max(1, numBins) * static_cast<int>(sizeof(int)), windowLens.data(), GL_DYNAMIC_DRAW);

        unsigned int params[16] = {};
        params[2] = static_cast<unsigned int>(std::max(2, fftLength));
        params[3] = static_cast<unsigned int>(std::max(0, windowMode));
        params[4] = static_cast<unsigned int>(std::max(0, normalizationMode));
        bindBufferData(f, buffers_[6], sizeof(params), params, GL_DYNAMIC_DRAW);

        numBins_ = numBins;
        windowMode_ = windowMode;
        normalizationMode_ = normalizationMode;
        fftLength_ = fftLength;
        cachedSpectrum_.assign(numBins_, 0.0f);
        cachedPhase_.assign(numBins_, 0.0f);
        hasCachedSpectrum_ = false;
        activeOutputBufferIndex_ = 0;
        for (auto& fence : outputFences_) {
            if (fence) {
                f->glDeleteSync(fence);
                fence = nullptr;
            }
        }
        context_->doneCurrent();
        initialized_ = true;
        return true;
    }

    bool compute(const std::vector<float>& ring,
                 unsigned int offset,
                 unsigned int availableSamples,
                 std::uint64_t sampleCount,
                 float leakiness,
                 std::vector<float>& outSpectrum,
                 std::vector<float>* outPhase)
    {
        if (!initialized_ || !ensureContext() || numBins_ <= 0) return false;
        if (ring.size() != static_cast<size_t>(signalLength_)) return false;
        if (!context_->makeCurrent(surface_.get())) return false;
        auto* f = context_->extraFunctions();

        f->glUseProgram(program_);

        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[0]);
        f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, signalLength_ * static_cast<int>(sizeof(float)), ring.data());
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers_[0]);

        const int writeIndex = activeOutputBufferIndex_;
        const int readIndex = 1 - activeOutputBufferIndex_;
        const GLuint outputMagBuffer = writeIndex == 0 ? buffers_[1] : buffers_[7];
        const GLuint readMagBuffer = readIndex == 0 ? buffers_[1] : buffers_[7];
        const GLuint outputPhaseBuffer = writeIndex == 0 ? buffers_[2] : buffers_[8];
        const GLuint readPhaseBuffer = readIndex == 0 ? buffers_[2] : buffers_[8];

        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputMagBuffer);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, outputPhaseBuffer);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers_[3]);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, buffers_[4]);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, buffers_[5]);

        unsigned int params[16] = {};
        params[0] = offset;
        params[1] = std::min<unsigned int>(availableSamples, static_cast<unsigned int>(signalLength_));
        params[2] = static_cast<unsigned int>(std::max(2, fftLength_));
        params[3] = static_cast<unsigned int>(std::max(0, windowMode_));
        params[4] = static_cast<unsigned int>(std::max(0, normalizationMode_));
        params[6] = static_cast<unsigned int>(sampleCount & 0xffffffffu);
        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[6]);
        f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(params), params);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, buffers_[6]);

        const GLint leakLoc = f->glGetUniformLocation(program_, "leakiness");
        if (leakLoc >= 0) {
            f->glUniform1f(leakLoc, leakiness);
        }

        f->glDispatchCompute(static_cast<GLuint>(numBins_), 1, 1);
        f->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        if (outputFences_[writeIndex]) {
            f->glDeleteSync(outputFences_[writeIndex]);
            outputFences_[writeIndex] = nullptr;
        }
        outputFences_[writeIndex] = f->glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        activeOutputBufferIndex_ = readIndex;

        if (outputFences_[readIndex]) {
            const GLenum waitResult = f->glClientWaitSync(outputFences_[readIndex], 0, 0);
            if (waitResult == GL_ALREADY_SIGNALED || waitResult == GL_CONDITION_SATISFIED) {
                f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, readMagBuffer);
                void* mapped = f->glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                                   0,
                                                   numBins_ * static_cast<int>(sizeof(float)),
                                                   GL_MAP_READ_BIT);
                if (mapped) {
                    cachedSpectrum_.resize(static_cast<size_t>(numBins_));
                    std::memcpy(cachedSpectrum_.data(), mapped, static_cast<size_t>(numBins_) * sizeof(float));
                    f->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                    f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, readPhaseBuffer);
                    void* mappedPhase = f->glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                                            0,
                                                            numBins_ * static_cast<int>(sizeof(float)),
                                                            GL_MAP_READ_BIT);
                    if (mappedPhase) {
                        cachedPhase_.resize(static_cast<size_t>(numBins_));
                        std::memcpy(cachedPhase_.data(), mappedPhase, static_cast<size_t>(numBins_) * sizeof(float));
                        f->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                    } else {
                        cachedPhase_.assign(static_cast<size_t>(numBins_), 0.0f);
                    }
                    hasCachedSpectrum_ = true;
                }
                f->glDeleteSync(outputFences_[readIndex]);
                outputFences_[readIndex] = nullptr;
            }
        }

        if (hasCachedSpectrum_) {
            outSpectrum = cachedSpectrum_;
            if (outPhase) {
                *outPhase = cachedPhase_;
            }
            context_->doneCurrent();
            return true;
        }

        context_->doneCurrent();
        return false;
    }

private:
    bool ensureContext()
    {
        if (context_ && surface_) return true;

        QSurfaceFormat format;
        format.setMajorVersion(4);
        format.setMinorVersion(3);
        format.setProfile(QSurfaceFormat::CoreProfile);

        surface_ = std::make_unique<QOffscreenSurface>();
        surface_->setFormat(format);
        surface_->create();
        if (!surface_->isValid()) return false;

        context_ = std::make_unique<QOpenGLContext>();
        context_->setFormat(format);
        if (!context_->create()) return false;
        return true;
    }

    static void bindBufferData(QOpenGLExtraFunctions* f, GLuint buffer, int size, const void* data, GLenum usage)
    {
        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        f->glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, usage);
    }

    std::unique_ptr<QOffscreenSurface> surface_;
    std::unique_ptr<QOpenGLContext> context_;
    GLuint program_ = 0;
    GLuint buffers_[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    bool buffersInitialized_ = false;
    bool initialized_ = false;
    int signalLength_ = 0;
    int numBins_ = 0;
    int algorithmMode_ = 0;
    int windowMode_ = 0;
    int normalizationMode_ = 0;
    int fftLength_ = 2;
    GLsync outputFences_[2] = {nullptr, nullptr};
    int activeOutputBufferIndex_ = 0;
    std::vector<float> cachedSpectrum_;
    std::vector<float> cachedPhase_;
    bool hasCachedSpectrum_ = false;
};

LoiaconoGpuCompute::LoiaconoGpuCompute()
    : impl_(std::make_unique<Impl>())
{
}

LoiaconoGpuCompute::~LoiaconoGpuCompute() = default;

bool LoiaconoGpuCompute::available() const
{
    return impl_->available();
}

bool LoiaconoGpuCompute::configure(int signalLength,
                                   int numBins,
                                   const std::vector<double>& freqs,
                                   const std::vector<double>& norms,
                                   const std::vector<int>& windowLens,
                                   int algorithmMode,
                                   int windowMode,
                                   int normalizationMode,
                                   int fftLength)
{
    return impl_->configure(signalLength,
                            numBins,
                            freqs,
                            norms,
                            windowLens,
                            algorithmMode,
                            windowMode,
                            normalizationMode,
                            fftLength);
}

bool LoiaconoGpuCompute::compute(const std::vector<float>& ring,
                                 unsigned int offset,
                                 unsigned int availableSamples,
                                 std::uint64_t sampleCount,
                                 float leakiness,
                                 std::vector<float>& outSpectrum,
                                 std::vector<float>* outPhase)
{
    return impl_->compute(ring, offset, availableSamples, sampleCount, leakiness, outSpectrum, outPhase);
}
