#include "audio_device_labels.h"

bool isDesktopAudioDeviceName(const QString& name)
{
    QString lower = name.toLower();
    return lower.contains("monitor of")
        || lower.contains(".monitor")
        || lower.contains("loopback")
        || lower.contains("stereo mix")
        || lower.contains("what u hear")
        || lower.contains("wave out")
        || lower.contains("monitor source")
        || lower.contains("output monitor");
}

bool isDesktopAudioDevice(const QString& backendName, const QString& rawName,
                          unsigned int inputChannels, unsigned int outputChannels)
{
    if (isDesktopAudioDeviceName(rawName)) {
        return true;
    }

    const QString backend = backendName.trimmed().toLower();
    const bool isPulseLike = backend.contains("pulse") || backend.contains("pipewire");
    return isPulseLike && inputChannels > 0 && outputChannels > 0;
}

QString desktopAudioLabel(const QString& rawName)
{
    QString name = rawName.trimmed();
    QString lower = name.toLower();
    if (lower.startsWith("monitor of ")) {
        name = name.mid(QStringLiteral("Monitor of ").size()).trimmed();
    }
    if (lower.endsWith(".monitor")) {
        name.chop(QStringLiteral(".monitor").size());
        name = name.trimmed();
    }
    if (name.isEmpty()) {
        return "Desktop audio";
    }
    return QString("Desktop audio: %1").arg(name);
}

QString displayNameForDevice(const QString& backendName, const QString& rawName,
                             unsigned int inputChannels, unsigned int outputChannels)
{
    if (!isDesktopAudioDevice(backendName, rawName, inputChannels, outputChannels)) {
        return rawName;
    }

    QString label = desktopAudioLabel(rawName);
    const QString backend = backendName.trimmed();
    if (!backend.isEmpty()) {
        label += QString(" (%1)").arg(backend);
    }
    return label;
}

QString displayNameForDeviceName(const QString& rawName)
{
    if (isDesktopAudioDeviceName(rawName)) {
        return desktopAudioLabel(rawName);
    }
    return rawName;
}
