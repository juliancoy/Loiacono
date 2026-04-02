#include <iostream>

#include "RtAudio.h"
#include "../audio_device_labels.h"

int main()
{
    std::cout << "Audio device dump" << std::endl;
    std::cout << "=================" << std::endl;
    std::vector<RtAudio::Api> apis;
    RtAudio::getCompiledApi(apis);

    for (auto api : apis) {
        RtAudio adc(api);
        auto ids = adc.getDeviceIds();
        unsigned int defaultIn = adc.getDefaultInputDevice();

        std::cout << "\nAPI: " << RtAudio::getApiDisplayName(api) << std::endl;
        std::cout << "Default input id: " << defaultIn << std::endl;
        std::cout << "Device count: " << ids.size() << std::endl;

        for (auto id : ids) {
            auto info = adc.getDeviceInfo(id);
            QString backendName = QString::fromStdString(RtAudio::getApiDisplayName(api));
            QString rawName = QString::fromStdString(info.name);
            std::cout << "\nID: " << id << std::endl;
            std::cout << "  Raw name: " << rawName.toStdString() << std::endl;
            std::cout << "  Display name: "
                      << displayNameForDevice(backendName, rawName,
                                              info.inputChannels, info.outputChannels).toStdString()
                      << std::endl;
            std::cout << "  Input channels: " << info.inputChannels << std::endl;
            std::cout << "  Output channels: " << info.outputChannels << std::endl;
            std::cout << "  Preferred sample rate: " << info.preferredSampleRate << std::endl;
            std::cout << "  Default input: " << (id == defaultIn ? "yes" : "no") << std::endl;
            std::cout << "  Desktop audio detected: "
                      << (isDesktopAudioDevice(backendName, rawName,
                                               info.inputChannels, info.outputChannels) ? "yes" : "no")
                      << std::endl;
        }
    }

    return 0;
}
