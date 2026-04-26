#include "loiacono_vulkan_compute.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QProcess>
#include <QString>
#include <QStringList>
#include <QTemporaryDir>

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

namespace {
constexpr uint32_t THREADS_PER_WORKGROUP = 128;

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
        "layout(std430, set = 0, binding = 0) buffer x_buf { readonly float x[SIGNAL_LENGTH]; };\n"
        "layout(std430, set = 0, binding = 1) buffer L_buf { writeonly float L[]; };\n"
        "layout(std430, set = 0, binding = 2) buffer f_buf { readonly float f[]; };\n"
        "layout(std430, set = 0, binding = 3) buffer norm_buf { readonly float norm[]; };\n"
        "layout(std430, set = 0, binding = 4) buffer window_buf { readonly int windowLen[]; };\n"
        "layout(std430, set = 0, binding = 5) buffer params_buf { readonly uint params[16]; };\n";

    source.replace("DEFINE_STRING", defines);
    source.replace("BUFFERS_STRING", buffers);
    source.replace("LEAKINESS_DECL", "layout(push_constant) uniform Params { float leakiness; } paramsBlock;\n");
    source.replace("LEAKINESS_VALUE", "paramsBlock.leakiness");
    return source;
}

struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void* mapped = nullptr;
    VkDeviceSize size = 0;
};

bool check(VkResult result)
{
    return result == VK_SUCCESS;
}
}

class LoiaconoVulkanCompute::Impl {
public:
    ~Impl()
    {
        destroy();
    }

    bool available() const
    {
        return initialized_;
    }

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
        if (!ensureDevice()) return false;
        if (!ensurePipeline(signalLength, algorithmMode)) return false;
        if (!ensureBuffers(signalLength, numBins)) return false;

        signalLength_ = signalLength;
        numBins_ = numBins;
        algorithmMode_ = algorithmMode;
        windowMode_ = windowMode;
        normalizationMode_ = normalizationMode;
        fftLength_ = fftLength;

        std::vector<float> freqFloats(numBins);
        std::vector<float> normFloats(numBins);
        for (int i = 0; i < numBins; ++i) {
            freqFloats[static_cast<size_t>(i)] = static_cast<float>(freqs[static_cast<size_t>(i)]);
            normFloats[static_cast<size_t>(i)] = static_cast<float>(norms[static_cast<size_t>(i)]);
        }

        std::memcpy(freqBuffer_.mapped, freqFloats.data(), static_cast<size_t>(numBins) * sizeof(float));
        std::memcpy(normBuffer_.mapped, normFloats.data(), static_cast<size_t>(numBins) * sizeof(float));
        std::memcpy(windowBuffer_.mapped, windowLens.data(), static_cast<size_t>(numBins) * sizeof(int));
        std::array<uint32_t, 16> params{};
        params[2] = static_cast<uint32_t>(std::max(2, fftLength_));
        params[3] = static_cast<uint32_t>(std::max(0, windowMode_));
        params[4] = static_cast<uint32_t>(std::max(0, normalizationMode_));
        std::memcpy(paramsBuffer_.mapped, params.data(), sizeof(params));

        initialized_ = true;
        return true;
    }

    bool compute(const std::vector<float>& ring,
                 unsigned int offset,
                 unsigned int availableSamples,
                 float leakiness,
                 std::vector<float>& outSpectrum)
    {
        if (!initialized_ || ring.size() != static_cast<size_t>(signalLength_)) return false;

        std::memcpy(signalBuffer_.mapped, ring.data(), static_cast<size_t>(signalLength_) * sizeof(float));
        std::array<uint32_t, 16> params{};
        params[0] = offset;
        params[1] = std::min<uint32_t>(availableSamples, static_cast<uint32_t>(signalLength_));
        params[2] = static_cast<uint32_t>(std::max(2, fftLength_));
        params[3] = static_cast<uint32_t>(std::max(0, windowMode_));
        params[4] = static_cast<uint32_t>(std::max(0, normalizationMode_));
        std::memcpy(paramsBuffer_.mapped, params.data(), sizeof(params));

        if (!recordAndSubmit(leakiness)) return false;

        outSpectrum.resize(static_cast<size_t>(numBins_));
        std::memcpy(outSpectrum.data(), outputBuffer_.mapped, static_cast<size_t>(numBins_) * sizeof(float));
        return true;
    }

private:
    bool ensureDevice()
    {
        if (device_ != VK_NULL_HANDLE) return true;

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Loiacono Spectrogram";
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo instanceInfo{};
        instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instanceInfo.pApplicationInfo = &appInfo;
        if (!check(vkCreateInstance(&instanceInfo, nullptr, &instance_))) return false;

        uint32_t physicalCount = 0;
        if (!check(vkEnumeratePhysicalDevices(instance_, &physicalCount, nullptr)) || physicalCount == 0) return false;
        std::vector<VkPhysicalDevice> physicalDevices(physicalCount);
        if (!check(vkEnumeratePhysicalDevices(instance_, &physicalCount, physicalDevices.data()))) return false;

        for (VkPhysicalDevice candidate : physicalDevices) {
            uint32_t familyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, nullptr);
            std::vector<VkQueueFamilyProperties> families(familyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(candidate, &familyCount, families.data());
            for (uint32_t i = 0; i < familyCount; ++i) {
                if (families[static_cast<size_t>(i)].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    physicalDevice_ = candidate;
                    queueFamilyIndex_ = i;
                    break;
                }
            }
            if (physicalDevice_ != VK_NULL_HANDLE) break;
        }
        if (physicalDevice_ == VK_NULL_HANDLE) return false;

        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo queueInfo{};
        queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueInfo.queueFamilyIndex = queueFamilyIndex_;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo deviceInfo{};
        deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        if (!check(vkCreateDevice(physicalDevice_, &deviceInfo, nullptr, &device_))) return false;

        vkGetDeviceQueue(device_, queueFamilyIndex_, 0, &queue_);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = queueFamilyIndex_;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (!check(vkCreateCommandPool(device_, &poolInfo, nullptr, &commandPool_))) return false;

        VkCommandBufferAllocateInfo cmdAlloc{};
        cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAlloc.commandPool = commandPool_;
        cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAlloc.commandBufferCount = 1;
        if (!check(vkAllocateCommandBuffers(device_, &cmdAlloc, &commandBuffer_))) return false;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (!check(vkCreateFence(device_, &fenceInfo, nullptr, &fence_))) return false;

        return true;
    }

    bool ensurePipeline(int signalLength, int algorithmMode)
    {
        if (pipeline_ != VK_NULL_HANDLE && signalLength == signalLength_ && algorithmMode == algorithmMode_) {
            return true;
        }

        destroyPipeline();

        QByteArray spirv = compileShader(signalLength, algorithmMode);
        if (spirv.isEmpty()) return false;

        VkShaderModuleCreateInfo shaderInfo{};
        shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderInfo.codeSize = static_cast<size_t>(spirv.size());
        shaderInfo.pCode = reinterpret_cast<const uint32_t*>(spirv.constData());
        if (!check(vkCreateShaderModule(device_, &shaderInfo, nullptr, &shaderModule_))) return false;

        std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
        for (uint32_t i = 0; i < bindings.size(); ++i) {
            bindings[static_cast<size_t>(i)].binding = i;
            bindings[static_cast<size_t>(i)].descriptorCount = 1;
            bindings[static_cast<size_t>(i)].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[static_cast<size_t>(i)].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }

        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();
        if (!check(vkCreateDescriptorSetLayout(device_, &layoutInfo, nullptr, &descriptorSetLayout_))) return false;

        VkPushConstantRange pushRange{};
        pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushRange.offset = 0;
        pushRange.size = sizeof(float);

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushRange;
        if (!check(vkCreatePipelineLayout(device_, &pipelineLayoutInfo, nullptr, &pipelineLayout_))) return false;

        VkPipelineShaderStageCreateInfo stageInfo{};
        stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = shaderModule_;
        stageInfo.pName = "main";

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = stageInfo;
        pipelineInfo.layout = pipelineLayout_;
        if (!check(vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline_))) return false;

        std::array<VkDescriptorPoolSize, 1> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSizes[0].descriptorCount = 6;
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = poolSizes.data();
        if (!check(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &descriptorPool_))) return false;

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool_;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &descriptorSetLayout_;
        if (!check(vkAllocateDescriptorSets(device_, &allocInfo, &descriptorSet_))) return false;

        return true;
    }

    bool ensureBuffers(int signalLength, int numBins)
    {
        if (signalBuffer_.buffer != VK_NULL_HANDLE && signalLength == signalLength_ && numBins == numBins_) {
            return true;
        }

        destroyBuffers();

        if (!createBuffer(signalLength * sizeof(float), signalBuffer_)) return false;
        if (!createBuffer(std::max(1, numBins) * static_cast<int>(sizeof(float)), outputBuffer_)) return false;
        if (!createBuffer(std::max(1, numBins) * static_cast<int>(sizeof(float)), freqBuffer_)) return false;
        if (!createBuffer(std::max(1, numBins) * static_cast<int>(sizeof(float)), normBuffer_)) return false;
        if (!createBuffer(std::max(1, numBins) * static_cast<int>(sizeof(int)), windowBuffer_)) return false;
        if (!createBuffer(16 * sizeof(uint32_t), paramsBuffer_)) return false;

        std::array<VkWriteDescriptorSet, 6> writes{};
        std::array<VkDescriptorBufferInfo, 6> infos{};
        const Buffer* buffers[] = {&signalBuffer_, &outputBuffer_, &freqBuffer_, &normBuffer_, &windowBuffer_, &paramsBuffer_};
        for (uint32_t i = 0; i < infos.size(); ++i) {
            infos[static_cast<size_t>(i)].buffer = buffers[i]->buffer;
            infos[static_cast<size_t>(i)].offset = 0;
            infos[static_cast<size_t>(i)].range = buffers[i]->size;
            writes[static_cast<size_t>(i)].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[static_cast<size_t>(i)].dstSet = descriptorSet_;
            writes[static_cast<size_t>(i)].dstBinding = i;
            writes[static_cast<size_t>(i)].descriptorCount = 1;
            writes[static_cast<size_t>(i)].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[static_cast<size_t>(i)].pBufferInfo = &infos[static_cast<size_t>(i)];
        }
        vkUpdateDescriptorSets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        return true;
    }

    bool recordAndSubmit(float leakiness)
    {
        vkResetFences(device_, 1, &fence_);
        vkResetCommandBuffer(commandBuffer_, 0);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        if (!check(vkBeginCommandBuffer(commandBuffer_, &beginInfo))) return false;

        vkCmdBindPipeline(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        vkCmdBindDescriptorSets(commandBuffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout_, 0, 1, &descriptorSet_, 0, nullptr);
        vkCmdPushConstants(commandBuffer_, pipelineLayout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(float), &leakiness);
        vkCmdDispatch(commandBuffer_, static_cast<uint32_t>(numBins_), 1, 1);

        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = outputBuffer_.buffer;
        barrier.offset = 0;
        barrier.size = outputBuffer_.size;
        vkCmdPipelineBarrier(commandBuffer_,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             0, nullptr,
                             1, &barrier,
                             0, nullptr);

        if (!check(vkEndCommandBuffer(commandBuffer_))) return false;

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer_;
        if (!check(vkQueueSubmit(queue_, 1, &submitInfo, fence_))) return false;
        if (!check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX))) return false;
        return true;
    }

    bool createBuffer(VkDeviceSize size, Buffer& out)
    {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!check(vkCreateBuffer(device_, &bufferInfo, nullptr, &out.buffer))) return false;

        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, out.buffer, &req);

        VkPhysicalDeviceMemoryProperties memProps{};
        vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
        uint32_t memoryTypeIndex = UINT32_MAX;
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            const bool supported = req.memoryTypeBits & (1u << i);
            const auto flags = memProps.memoryTypes[i].propertyFlags;
            const bool wanted = (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
                == (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (supported && wanted) {
                memoryTypeIndex = i;
                break;
            }
        }
        if (memoryTypeIndex == UINT32_MAX) return false;

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = req.size;
        allocInfo.memoryTypeIndex = memoryTypeIndex;
        if (!check(vkAllocateMemory(device_, &allocInfo, nullptr, &out.memory))) return false;
        if (!check(vkBindBufferMemory(device_, out.buffer, out.memory, 0))) return false;
        if (!check(vkMapMemory(device_, out.memory, 0, req.size, 0, &out.mapped))) return false;
        out.size = size;
        return true;
    }

    QByteArray compileShader(int signalLength, int algorithmMode)
    {
        const QString source = buildShaderSource(signalLength, algorithmMode);
        if (source.isEmpty()) return {};

        QTemporaryDir dir;
        if (!dir.isValid()) return {};

        const QString srcPath = dir.filePath(QStringLiteral("loiacono_vulkan.comp"));
        const QString spvPath = dir.filePath(QStringLiteral("loiacono_vulkan.spv"));
        QFile srcFile(srcPath);
        if (!srcFile.open(QIODevice::WriteOnly | QIODevice::Text)) return {};
        srcFile.write(source.toUtf8());
        srcFile.close();

        QProcess proc;
        proc.start(QStringLiteral("glslangValidator"), {QStringLiteral("-V"), srcPath, QStringLiteral("-o"), spvPath});
        if (!proc.waitForFinished(10000) || proc.exitStatus() != QProcess::NormalExit || proc.exitCode() != 0) {
            return {};
        }

        QFile spvFile(spvPath);
        if (!spvFile.open(QIODevice::ReadOnly)) return {};
        return spvFile.readAll();
    }

    void destroyBuffer(Buffer& buffer)
    {
        if (buffer.mapped && device_ != VK_NULL_HANDLE) {
            vkUnmapMemory(device_, buffer.memory);
        }
        if (buffer.buffer != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer.buffer, nullptr);
        }
        if (buffer.memory != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, buffer.memory, nullptr);
        }
        buffer = {};
    }

    void destroyBuffers()
    {
        destroyBuffer(signalBuffer_);
        destroyBuffer(outputBuffer_);
        destroyBuffer(freqBuffer_);
        destroyBuffer(normBuffer_);
        destroyBuffer(windowBuffer_);
        destroyBuffer(paramsBuffer_);
    }

    void destroyPipeline()
    {
        if (descriptorPool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, descriptorPool_, nullptr);
            descriptorPool_ = VK_NULL_HANDLE;
        }
        if (pipeline_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
            pipeline_ = VK_NULL_HANDLE;
        }
        if (pipelineLayout_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            pipelineLayout_ = VK_NULL_HANDLE;
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
            descriptorSetLayout_ = VK_NULL_HANDLE;
        }
        if (shaderModule_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, shaderModule_, nullptr);
            shaderModule_ = VK_NULL_HANDLE;
        }
    }

    void destroy()
    {
        destroyBuffers();
        destroyPipeline();
        if (fence_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyFence(device_, fence_, nullptr);
            fence_ = VK_NULL_HANDLE;
        }
        if (commandPool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE) {
            vkDestroyCommandPool(device_, commandPool_, nullptr);
            commandPool_ = VK_NULL_HANDLE;
        }
        if (device_ != VK_NULL_HANDLE) {
            vkDestroyDevice(device_, nullptr);
            device_ = VK_NULL_HANDLE;
        }
        if (instance_ != VK_NULL_HANDLE) {
            vkDestroyInstance(instance_, nullptr);
            instance_ = VK_NULL_HANDLE;
        }
        physicalDevice_ = VK_NULL_HANDLE;
        queue_ = VK_NULL_HANDLE;
        initialized_ = false;
    }

    VkInstance instance_ = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamilyIndex_ = 0;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkShaderModule shaderModule_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet_ = VK_NULL_HANDLE;

    Buffer signalBuffer_;
    Buffer outputBuffer_;
    Buffer freqBuffer_;
    Buffer normBuffer_;
    Buffer windowBuffer_;
    Buffer paramsBuffer_;

    int signalLength_ = 0;
    int numBins_ = 0;
    int algorithmMode_ = 0;
    int windowMode_ = 0;
    int normalizationMode_ = 0;
    int fftLength_ = 2;
    bool initialized_ = false;
};

LoiaconoVulkanCompute::LoiaconoVulkanCompute()
    : impl_(std::make_unique<Impl>())
{
}

LoiaconoVulkanCompute::~LoiaconoVulkanCompute() = default;

bool LoiaconoVulkanCompute::available() const
{
    return impl_->available();
}

bool LoiaconoVulkanCompute::configure(int signalLength,
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

bool LoiaconoVulkanCompute::compute(const std::vector<float>& ring,
                                    unsigned int offset,
                                    unsigned int availableSamples,
                                    float leakiness,
                                    std::vector<float>& outSpectrum)
{
    return impl_->compute(ring, offset, availableSamples, leakiness, outSpectrum);
}
