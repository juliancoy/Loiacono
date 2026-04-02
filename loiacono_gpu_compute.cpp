#include "loiacono_gpu_compute.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QSurfaceFormat>

#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr int THREADS_PER_WORKGROUP = 128;

QString loadShaderTemplate()
{
    const QString path = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath("../../shaders/loiacono_generic.comp.template");
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return {};
    return QString::fromUtf8(file.readAll());
}

QString buildShaderSource(int signalLength)
{
    QString source = loadShaderTemplate();
    if (source.isEmpty()) return {};

    const QString defines = QString(
        "#define PI 3.1415926\n"
        "#define SIGNAL_LENGTH %1\n"
        "#define THREADS_PER_WORKGROUP %2\n")
        .arg(signalLength)
        .arg(THREADS_PER_WORKGROUP);

    const QString buffers =
        "layout(std430, binding = 0) buffer x_buf { readonly float x[SIGNAL_LENGTH]; };\n"
        "layout(std430, binding = 1) buffer L_buf { writeonly float L[]; };\n"
        "layout(std430, binding = 2) buffer f_buf { readonly float f[]; };\n"
        "layout(std430, binding = 3) buffer norm_buf { readonly float norm[]; };\n"
        "layout(std430, binding = 4) buffer window_buf { readonly int windowLen[]; };\n"
        "layout(std430, binding = 5) buffer offset_buf { readonly uint offset[16]; };\n";

    source.replace("DEFINE_STRING", defines);
    source.replace("BUFFERS_STRING", buffers);
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
        if (buffers_[0]) f->glDeleteBuffers(7, buffers_);
        context_->doneCurrent();
    }

    bool available() const { return initialized_; }

    bool configure(int signalLength,
                   int numBins,
                   const std::vector<double>& freqs,
                   const std::vector<double>& norms,
                   const std::vector<int>& windowLens)
    {
        if (!ensureContext()) return false;
        if (!context_->makeCurrent(surface_.get())) return false;
        auto* f = context_->extraFunctions();

        if (signalLength_ != signalLength || program_ == 0) {
            if (program_) {
                f->glDeleteProgram(program_);
                program_ = 0;
            }

            QString source = buildShaderSource(signalLength);
            if (source.isEmpty()) {
                context_->doneCurrent();
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
                return false;
            }

            signalLength_ = signalLength;
        }

        if (!buffersInitialized_) {
            f->glGenBuffers(7, buffers_);
            buffersInitialized_ = true;
        }

        std::vector<float> freqFloats(numBins);
        std::vector<float> normFloats(numBins);
        for (int i = 0; i < numBins; ++i) {
            freqFloats[i] = static_cast<float>(freqs[i]);
            normFloats[i] = static_cast<float>(norms[i]);
        }

        bindBufferData(f, buffers_[0], signalLength * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[1], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[6], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[2], std::max(1, numBins) * static_cast<int>(sizeof(float)), freqFloats.data(), GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[3], std::max(1, numBins) * static_cast<int>(sizeof(float)), normFloats.data(), GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[4], std::max(1, numBins) * static_cast<int>(sizeof(int)), windowLens.data(), GL_DYNAMIC_DRAW);
        unsigned int zeroOffset[16] = {};
        bindBufferData(f, buffers_[5], sizeof(zeroOffset), zeroOffset, GL_DYNAMIC_DRAW);

        numBins_ = numBins;
        cachedSpectrum_.assign(numBins_, 0.0f);
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

    bool compute(const std::vector<float>& ring, unsigned int offset, std::vector<float>& outSpectrum)
    {
        return compute(ring, offset, 1.0f, outSpectrum);
    }

    bool compute(const std::vector<float>& ring, unsigned int offset, float leakiness, std::vector<float>& outSpectrum)
    {
        if (!initialized_ || !ensureContext() || numBins_ <= 0) return false;
        if (!context_->makeCurrent(surface_.get())) return false;
        auto* f = context_->extraFunctions();

        f->glUseProgram(program_);

        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[0]);
        f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, signalLength_ * static_cast<int>(sizeof(float)), ring.data());
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffers_[0]);

        const int writeIndex = activeOutputBufferIndex_;
        const int readIndex = 1 - activeOutputBufferIndex_;
        const GLuint outputBuffer = writeIndex == 0 ? buffers_[1] : buffers_[6];
        const GLuint readBuffer = readIndex == 0 ? buffers_[1] : buffers_[6];

        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, outputBuffer);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, buffers_[2]);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, buffers_[3]);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, buffers_[4]);

        unsigned int offsets[16] = {};
        offsets[0] = offset;
        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[5]);
        f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(offsets), offsets);
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, buffers_[5]);

        GLint leakLoc = f->glGetUniformLocation(program_, "leakiness");
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

        bool updatedCache = false;
        if (outputFences_[readIndex]) {
            GLenum waitResult = f->glClientWaitSync(outputFences_[readIndex], 0, 0);
            if (waitResult == GL_ALREADY_SIGNALED || waitResult == GL_CONDITION_SATISFIED) {
                f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, readBuffer);
                void* mapped = f->glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0,
                                                   numBins_ * static_cast<int>(sizeof(float)),
                                                   GL_MAP_READ_BIT);
                if (mapped) {
                    cachedSpectrum_.resize(numBins_);
                    std::memcpy(cachedSpectrum_.data(), mapped, numBins_ * sizeof(float));
                    f->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
                    hasCachedSpectrum_ = true;
                    updatedCache = true;
                }
                f->glDeleteSync(outputFences_[readIndex]);
                outputFences_[readIndex] = nullptr;
            }
        }

        if (hasCachedSpectrum_) {
            outSpectrum = cachedSpectrum_;
            context_->doneCurrent();
            return true;
        }

        if (updatedCache) {
            outSpectrum = cachedSpectrum_;
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
    GLuint buffers_[7] = {0, 0, 0, 0, 0, 0, 0};
    bool buffersInitialized_ = false;
    bool initialized_ = false;
    int signalLength_ = 0;
    int numBins_ = 0;
    GLsync outputFences_[2] = {nullptr, nullptr};
    int activeOutputBufferIndex_ = 0;
    std::vector<float> cachedSpectrum_;
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
                                   const std::vector<int>& windowLens)
{
    return impl_->configure(signalLength, numBins, freqs, norms, windowLens);
}

bool LoiaconoGpuCompute::compute(const std::vector<float>& ring, unsigned int offset, float leakiness, std::vector<float>& outSpectrum)
{
    return impl_->compute(ring, offset, leakiness, outSpectrum);
}
