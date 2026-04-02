#pragma once

#include <QString>

bool isDesktopAudioDeviceName(const QString& name);
bool isDesktopAudioDevice(const QString& backendName, const QString& rawName,
                          unsigned int inputChannels, unsigned int outputChannels);
QString desktopAudioLabel(const QString& rawName);
QString displayNameForDevice(const QString& backendName, const QString& rawName,
                             unsigned int inputChannels, unsigned int outputChannels);
QString displayNameForDeviceName(const QString& rawName);
