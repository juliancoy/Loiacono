#include <iostream>
#include <vector>

#include <QString>

#include "../audio_device_labels.h"

struct LabelCase {
    const char* backendName;
    const char* rawName;
    unsigned int inputChannels;
    unsigned int outputChannels;
    bool expectDesktopAudio;
    const char* expectedLabel;
};

int main()
{
    const std::vector<LabelCase> cases = {
        {"Pulse", "Monitor of Built-in Audio Analog Stereo", 2, 2, true, "Desktop audio: Built-in Audio Analog Stereo (Pulse)"},
        {"Pulse", "alsa_output.pci-0000_00_1f.3.analog-stereo.monitor", 2, 2, true, "Desktop audio: alsa_output.pci-0000_00_1f.3.analog-stereo (Pulse)"},
        {"Pulse", "Monitor Source of Starship/Matisse HD Audio Controller", 2, 2, true, "Desktop audio: Monitor Source of Starship/Matisse HD Audio Controller (Pulse)"},
        {"WASAPI", "Stereo Mix (Realtek Audio)", 2, 0, true, "Desktop audio: Stereo Mix (Realtek Audio) (WASAPI)"},
        {"ALSA", "Loopback: PCM (hw:0,0)", 2, 0, true, "Desktop audio: Loopback: PCM (hw:0,0) (ALSA)"},
        {"Pulse", "Starship/Matisse HD Audio Controller", 2, 2, true, "Desktop audio: Starship/Matisse HD Audio Controller (Pulse)"},
        {"Pulse", "罗技高清网络摄像机 C930c", 2, 0, false, "罗技高清网络摄像机 C930c"},
        {"ALSA", "USB Audio Device Microphone", 1, 0, false, "USB Audio Device Microphone"},
        {"ALSA", "Built-in Audio Analog Stereo", 2, 2, false, "Built-in Audio Analog Stereo"},
    };

    bool passed = true;
    for (const auto& testCase : cases) {
        QString rawName = QString::fromUtf8(testCase.rawName);
        QString backendName = QString::fromUtf8(testCase.backendName);
        bool detected = isDesktopAudioDevice(backendName, rawName,
                                             testCase.inputChannels,
                                             testCase.outputChannels);
        QString label = displayNameForDevice(backendName, rawName,
                                             testCase.inputChannels,
                                             testCase.outputChannels);

        std::cout << rawName.toStdString()
                  << " => detected=" << (detected ? "true" : "false")
                  << ", label=\"" << label.toStdString() << "\"";

        if (detected != testCase.expectDesktopAudio || label != QString::fromUtf8(testCase.expectedLabel)) {
            std::cout << " [FAIL]" << std::endl;
            passed = false;
        } else {
            std::cout << " [PASS]" << std::endl;
        }
    }

    if (!passed) {
        std::cout << "audio device label tests failed" << std::endl;
        return 1;
    }

    std::cout << "audio device label tests passed" << std::endl;
    return 0;
}
