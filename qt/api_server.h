#pragma once
#include <QTcpServer>
#include <QTcpSocket>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDir>
#include <QFile>
#include <QBuffer>
#include <QDateTime>
#include <QTimer>
#include <QSet>
#include <functional>

class LoiaconoRolling;
class SpectrogramWidget;

// Lightweight REST API server for the Loiacono spectrogram
//
// Endpoints:
//   GET  /api/version                - app version and build info
//   GET  /api/status                 - current peak freq, bins, sample rate
//   GET  /api/screenshot             - spectrogram as PNG image
//   GET  /api/profile                - current transform settings
//   PUT  /api/profile                - update transform settings
//   GET  /api/profiles               - list saved profile names
//   GET  /api/profiles/:name         - load a saved profile
//   POST /api/profiles/:name         - save current settings as named profile
//   DELETE /api/profiles/:name       - delete a saved profile
//   GET  /api/spectrum               - current spectrum as JSON array
//   GET  /api/stream                 - MJPEG video stream (use in <img> tag)
//   GET  /api/devices                - list audio input devices
//   PUT  /api/device                 - switch audio input device {"id": 131}

class ApiServer : public QTcpServer {
    Q_OBJECT
public:
    ApiServer(LoiaconoRolling* transform, SpectrogramWidget* spectrogram,
              QObject* parent = nullptr);

    bool startListening(quint16 port = 8080);

    // Called by main to sync slider state when API changes settings
    using SettingsCallback = std::function<void(int multiple, int bins, int freqMin, int freqMax)>;
    void setSettingsCallback(SettingsCallback cb) { settingsCallback_ = std::move(cb); }

    // Audio device callbacks
    using DeviceListCallback = std::function<QJsonArray()>;
    using DeviceSwitchCallback = std::function<QString(unsigned int deviceId)>;
    void setDeviceListCallback(DeviceListCallback cb) { deviceListCb_ = std::move(cb); }
    void setDeviceSwitchCallback(DeviceSwitchCallback cb) { deviceSwitchCb_ = std::move(cb); }

    // Keep current slider values in sync
    void updateCurrentSettings(int multiple, int bins, int freqMin, int freqMax);

protected:
    void incomingConnection(qintptr socketDescriptor) override;

private:
    void handleRequest(QTcpSocket* socket);
    void sendJson(QTcpSocket* socket, int status, const QJsonObject& obj);
    void sendJsonArray(QTcpSocket* socket, int status, const QJsonArray& arr);
    void sendPng(QTcpSocket* socket, const QByteArray& data);
    void sendHtml(QTcpSocket* socket, const QByteArray& html);
    void sendError(QTcpSocket* socket, int status, const QString& msg);
    void sendOk(QTcpSocket* socket, const QString& msg);
    void startMjpegStream(QTcpSocket* socket);
    void pushMjpegFrame();

    QString profileDir() const;

    LoiaconoRolling* transform_;
    SpectrogramWidget* spectrogram_;
    SettingsCallback settingsCallback_;
    DeviceListCallback deviceListCb_;
    DeviceSwitchCallback deviceSwitchCb_;
    QTimer streamTimer_;
    QSet<QTcpSocket*> streamClients_;

    // Cached current settings (kept in sync by main.cpp)
    int curMultiple_ = 40;
    int curBins_ = 200;
    int curFreqMin_ = 100;
    int curFreqMax_ = 3000;
};
