#include "loiacono_gpu_rolling_compute.h"

#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLExtraFunctions>
#include <QSurfaceFormat>

#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr int kThreadsPerGroup = 128;

const char* kRollingUpdateShader = R"(
    #version 430
    layout(local_size_x = 128) in;

    layout(std430, binding = 0) readonly buffer NewChunkBuf { float newChunk[]; };
    layout(std430, binding = 1) readonly buffer OldChunkBuf { float oldChunk[]; };
    layout(std430, binding = 2) readonly buffer FreqBuf { float freqs[]; };
    layout(std430, binding = 3) readonly buffer NormBuf { float norms[]; };
    layout(std430, binding = 4) readonly buffer WindowBuf { int windowLens[]; };
    layout(std430, binding = 5) buffer TrBuf { float trState[]; };
    layout(std430, binding = 6) buffer TiBuf { float tiState[]; };

    uniform int chunkLength;
    uniform int numBins;
    uniform uint sampleBaseLo;
    uniform uint sampleBaseHi;

    shared float sharedTr[128];
    shared float sharedTi[128];

    double sampleBase()
    {
        return double(sampleBaseHi) * 4294967296.0 + double(sampleBaseLo);
    }

    void main()
    {
        uint bin = gl_WorkGroupID.x;
        uint tid = gl_LocalInvocationID.x;
        if (bin >= uint(numBins)) return;

        float freq = freqs[bin];
        float norm = norms[bin];
        int windowLen = windowLens[bin];
        double base = sampleBase();

        float tr = 0.0;
        float ti = 0.0;
        for (int i = int(tid); i < chunkLength; i += 128) {
            double sampleIx = base + double(i);
            double angle = 6.283185307179586 * double(freq) * sampleIx;
            float c = float(cos(angle));
            float s = float(sin(angle));
            float sample = newChunk[i];
            tr += sample * c * norm;
            ti -= sample * s * norm;

            if (sampleIx >= double(windowLen)) {
                double oldAngle = 6.283185307179586 * double(freq) * (sampleIx - double(windowLen));
                float oldC = float(cos(oldAngle));
                float oldS = float(sin(oldAngle));
                float oldSample = oldChunk[i];
                tr -= oldSample * oldC * norm;
                ti += oldSample * oldS * norm;
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
            trState[bin] += sharedTr[0];
            tiState[bin] += sharedTi[0];
        }
    }
)";

const char* kMagnitudeShader = R"(
    #version 430
    layout(local_size_x = 128) in;

    layout(std430, binding = 0) readonly buffer TrBuf { float trState[]; };
    layout(std430, binding = 1) readonly buffer TiBuf { float tiState[]; };
    layout(std430, binding = 2) writeonly buffer MagBuf { float magnitude[]; };

    uniform int numBins;

    void main()
    {
        uint idx = gl_GlobalInvocationID.x;
        if (idx >= uint(numBins)) return;
        float tr = trState[idx];
        float ti = tiState[idx];
        magnitude[idx] = sqrt(tr * tr + ti * ti);
    }
)";
}

class LoiaconoGpuRollingCompute::Impl {
public:
    ~Impl()
    {
        if (!context_ || !surface_) return;
        context_->makeCurrent(surface_.get());
        auto* f = context_->extraFunctions();
        if (updateProgram_) f->glDeleteProgram(updateProgram_);
        if (magnitudeProgram_) f->glDeleteProgram(magnitudeProgram_);
        if (buffers_[0]) f->glDeleteBuffers(kBufferCount, buffers_);
        context_->doneCurrent();
    }

    bool available() const
    {
        return initialized_;
    }

    bool configure(int signalLength,
                   int maxChunkLength,
                   int numBins,
                   const std::vector<double>& freqs,
                   const std::vector<double>& norms,
                   const std::vector<int>& windowLens)
    {
        if (!ensureContext()) return false;
        if (!context_->makeCurrent(surface_.get())) return false;
        auto* f = context_->extraFunctions();

        if (!ensureProgram(f, updateProgram_, kRollingUpdateShader) ||
            !ensureProgram(f, magnitudeProgram_, kMagnitudeShader)) {
            context_->doneCurrent();
            return false;
        }

        if (!buffersInitialized_) {
            f->glGenBuffers(kBufferCount, buffers_);
            buffersInitialized_ = true;
        }

        const bool geometryChanged =
            signalLength_ != signalLength || maxChunkLength_ != maxChunkLength || numBins_ != numBins;

        if (geometryChanged) {
            bindBufferData(f, buffers_[0], std::max(1, maxChunkLength) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
            bindBufferData(f, buffers_[1], std::max(1, maxChunkLength) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
            bindBufferData(f, buffers_[5], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
            bindBufferData(f, buffers_[6], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);
            bindBufferData(f, buffers_[7], std::max(1, numBins) * static_cast<int>(sizeof(float)), nullptr, GL_DYNAMIC_DRAW);

            std::vector<float> zeros(std::max(1, numBins), 0.0f);
            f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[5]);
            f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numBins * static_cast<int>(sizeof(float)), zeros.data());
            f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[6]);
            f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, numBins * static_cast<int>(sizeof(float)), zeros.data());
        }

        std::vector<float> freqFloats(numBins);
        std::vector<float> normFloats(numBins);
        for (int i = 0; i < numBins; ++i) {
            freqFloats[i] = static_cast<float>(freqs[i]);
            normFloats[i] = static_cast<float>(norms[i]);
        }
        bindBufferData(f, buffers_[2], std::max(1, numBins) * static_cast<int>(sizeof(float)), freqFloats.data(), GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[3], std::max(1, numBins) * static_cast<int>(sizeof(float)), normFloats.data(), GL_DYNAMIC_DRAW);
        bindBufferData(f, buffers_[4], std::max(1, numBins) * static_cast<int>(sizeof(int)), windowLens.data(), GL_DYNAMIC_DRAW);

        signalLength_ = signalLength;
        maxChunkLength_ = maxChunkLength;
        numBins_ = numBins;
        initialized_ = true;
        context_->doneCurrent();
        return true;
    }

    bool processChunk(const float* newSamples,
                      const float* oldSamples,
                      int count,
                      std::uint64_t startSampleCount)
    {
        if (!initialized_ || count <= 0 || count > maxChunkLength_) return false;
        if (!context_->makeCurrent(surface_.get())) return false;
        auto* f = context_->extraFunctions();

        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[0]);
        f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * static_cast<int>(sizeof(float)), newSamples);
        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[1]);
        f->glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, count * static_cast<int>(sizeof(float)), oldSamples);

        f->glUseProgram(updateProgram_);
        bindBufferBase(f, 0, buffers_[0]);
        bindBufferBase(f, 1, buffers_[1]);
        bindBufferBase(f, 2, buffers_[2]);
        bindBufferBase(f, 3, buffers_[3]);
        bindBufferBase(f, 4, buffers_[4]);
        bindBufferBase(f, 5, buffers_[5]);
        bindBufferBase(f, 6, buffers_[6]);

        f->glUniform1i(f->glGetUniformLocation(updateProgram_, "chunkLength"), count);
        f->glUniform1i(f->glGetUniformLocation(updateProgram_, "numBins"), numBins_);
        f->glUniform1ui(f->glGetUniformLocation(updateProgram_, "sampleBaseLo"),
                        static_cast<GLuint>(startSampleCount & 0xffffffffu));
        f->glUniform1ui(f->glGetUniformLocation(updateProgram_, "sampleBaseHi"),
                        static_cast<GLuint>((startSampleCount >> 32) & 0xffffffffu));

        f->glDispatchCompute(static_cast<GLuint>(numBins_), 1, 1);
        f->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
        context_->doneCurrent();
        return true;
    }

    bool spectrum(std::vector<float>& outSpectrum) const
    {
        if (!initialized_ || numBins_ <= 0) return false;
        if (!context_->makeCurrent(surface_.get())) return false;
        auto* f = context_->extraFunctions();

        f->glUseProgram(magnitudeProgram_);
        bindBufferBase(f, 0, buffers_[5]);
        bindBufferBase(f, 1, buffers_[6]);
        bindBufferBase(f, 2, buffers_[7]);
        f->glUniform1i(f->glGetUniformLocation(magnitudeProgram_, "numBins"), numBins_);
        f->glDispatchCompute(static_cast<GLuint>((numBins_ + kThreadsPerGroup - 1) / kThreadsPerGroup), 1, 1);
        f->glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);

        outSpectrum.resize(numBins_);
        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffers_[7]);
        void* mapped = f->glMapBufferRange(GL_SHADER_STORAGE_BUFFER,
                                           0,
                                           numBins_ * static_cast<int>(sizeof(float)),
                                           GL_MAP_READ_BIT);
        if (!mapped) {
            context_->doneCurrent();
            return false;
        }
        std::memcpy(outSpectrum.data(), mapped, numBins_ * sizeof(float));
        f->glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        context_->doneCurrent();
        return true;
    }

private:
    static constexpr int kBufferCount = 8;

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

    static bool ensureProgram(QOpenGLExtraFunctions* f, GLuint& program, const char* source)
    {
        if (program) return true;

        GLuint shader = f->glCreateShader(GL_COMPUTE_SHADER);
        f->glShaderSource(shader, 1, &source, nullptr);
        f->glCompileShader(shader);

        GLint ok = GL_FALSE;
        f->glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            f->glDeleteShader(shader);
            return false;
        }

        program = f->glCreateProgram();
        f->glAttachShader(program, shader);
        f->glLinkProgram(program);
        f->glDeleteShader(shader);

        f->glGetProgramiv(program, GL_LINK_STATUS, &ok);
        if (!ok) {
            f->glDeleteProgram(program);
            program = 0;
            return false;
        }
        return true;
    }

    static void bindBufferData(QOpenGLExtraFunctions* f, GLuint buffer, int size, const void* data, GLenum usage)
    {
        f->glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer);
        f->glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, usage);
    }

    static void bindBufferBase(QOpenGLExtraFunctions* f, GLuint index, GLuint buffer)
    {
        f->glBindBufferBase(GL_SHADER_STORAGE_BUFFER, index, buffer);
    }

    mutable std::unique_ptr<QOffscreenSurface> surface_;
    mutable std::unique_ptr<QOpenGLContext> context_;
    GLuint updateProgram_ = 0;
    GLuint magnitudeProgram_ = 0;
    GLuint buffers_[kBufferCount] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool buffersInitialized_ = false;
    bool initialized_ = false;
    int signalLength_ = 0;
    int maxChunkLength_ = 0;
    int numBins_ = 0;
};

LoiaconoGpuRollingCompute::LoiaconoGpuRollingCompute()
    : impl_(std::make_unique<Impl>())
{
}

LoiaconoGpuRollingCompute::~LoiaconoGpuRollingCompute() = default;

bool LoiaconoGpuRollingCompute::available() const
{
    return impl_->available();
}

bool LoiaconoGpuRollingCompute::configure(int signalLength,
                                          int maxChunkLength,
                                          int numBins,
                                          const std::vector<double>& freqs,
                                          const std::vector<double>& norms,
                                          const std::vector<int>& windowLens)
{
    return impl_->configure(signalLength, maxChunkLength, numBins, freqs, norms, windowLens);
}

bool LoiaconoGpuRollingCompute::processChunk(const float* newSamples,
                                             const float* oldSamples,
                                             int count,
                                             std::uint64_t startSampleCount)
{
    return impl_->processChunk(newSamples, oldSamples, count, startSampleCount);
}

bool LoiaconoGpuRollingCompute::spectrum(std::vector<float>& outSpectrum) const
{
    return impl_->spectrum(outSpectrum);
}
