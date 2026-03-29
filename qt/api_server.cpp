#include "api_server.h"
#include "loiacono_rolling.h"
#include "spectrogram_widget.h"
#include <QImage>
#include <QStandardPaths>

static const char* APP_VERSION = "1.0.0";

ApiServer::ApiServer(LoiaconoRolling* transform, SpectrogramWidget* spectrogram,
                     QObject* parent)
    : QTcpServer(parent), transform_(transform), spectrogram_(spectrogram)
{
}

bool ApiServer::startListening(quint16 port)
{
    if (!listen(QHostAddress::LocalHost, port)) {
        qWarning("API server failed to listen on port %d: %s",
                 port, qPrintable(errorString()));
        return false;
    }
    qInfo("API server listening on http://localhost:%d", port);
    return true;
}

void ApiServer::updateCurrentSettings(int multiple, int bins, int freqMin, int freqMax)
{
    curMultiple_ = multiple;
    curBins_ = bins;
    curFreqMin_ = freqMin;
    curFreqMax_ = freqMax;
}

void ApiServer::incomingConnection(qintptr socketDescriptor)
{
    auto* socket = new QTcpSocket(this);
    socket->setSocketDescriptor(socketDescriptor);

    connect(socket, &QTcpSocket::readyRead, this, [this, socket]() {
        handleRequest(socket);
    });
    connect(socket, &QTcpSocket::disconnected, socket, &QTcpSocket::deleteLater);
}

void ApiServer::handleRequest(QTcpSocket* socket)
{
    auto data = socket->readAll();
    QString request = QString::fromUtf8(data);
    auto lines = request.split("\r\n");
    if (lines.isEmpty()) { sendError(socket, 400, "Bad request"); return; }

    auto parts = lines[0].split(" ");
    if (parts.size() < 2) { sendError(socket, 400, "Bad request"); return; }

    QString method = parts[0];
    QString path = parts[1];

    // Parse body for PUT/POST
    QJsonObject body;
    int bodyStart = request.indexOf("\r\n\r\n");
    if (bodyStart >= 0) {
        QByteArray bodyData = data.mid(bodyStart + 4);
        QJsonParseError err;
        auto doc = QJsonDocument::fromJson(bodyData, &err);
        if (doc.isObject()) body = doc.object();
    }

    // ── Route ──
    if (method == "GET" && path == "/api/version") {
        sendJson(socket, 200, {
            {"app", "Loiacono Spectrogram"},
            {"version", APP_VERSION},
            {"qt", qVersion()},
            {"platform", QSysInfo::prettyProductName()},
            {"arch", QSysInfo::currentCpuArchitecture()},
        });

    } else if (method == "GET" && path == "/api/status") {
        std::vector<float> spectrum;
        transform_->getSpectrum(spectrum);
        float peak = 0; int peakIdx = 0;
        for (int i = 0; i < (int)spectrum.size(); i++) {
            if (spectrum[i] > peak) { peak = spectrum[i]; peakIdx = i; }
        }
        double peakHz = transform_->numBins() > 0 ? transform_->binFreqHz(peakIdx) : 0;
        sendJson(socket, 200, {
            {"peakFrequencyHz", peakHz},
            {"peakAmplitude", static_cast<double>(peak)},
            {"numBins", transform_->numBins()},
            {"sampleRate", transform_->sampleRate()},
            {"multiple", curMultiple_},
            {"freqMin", curFreqMin_},
            {"freqMax", curFreqMax_},
        });

    } else if (method == "GET" && path == "/api/screenshot") {
        QImage img = spectrogram_->renderToImage();
        QByteArray pngData;
        QBuffer buf(&pngData);
        buf.open(QIODevice::WriteOnly);
        img.save(&buf, "PNG");
        sendPng(socket, pngData);

    } else if (method == "GET" && path == "/api/spectrum") {
        std::vector<float> spectrum;
        transform_->getSpectrum(spectrum);
        QJsonArray arr;
        for (int i = 0; i < (int)spectrum.size(); i++) {
            QJsonObject bin;
            bin["hz"] = transform_->binFreqHz(i);
            bin["amplitude"] = static_cast<double>(spectrum[i]);
            arr.append(bin);
        }
        sendJsonArray(socket, 200, arr);

    } else if (method == "GET" && path == "/api/profile") {
        sendJson(socket, 200, {
            {"multiple", curMultiple_},
            {"bins", curBins_},
            {"freqMin", curFreqMin_},
            {"freqMax", curFreqMax_},
            {"sampleRate", transform_->sampleRate()},
        });

    } else if (method == "PUT" && path == "/api/profile") {
        int m = body.value("multiple").toInt(curMultiple_);
        int b = body.value("bins").toInt(curBins_);
        int fmin = body.value("freqMin").toInt(curFreqMin_);
        int fmax = body.value("freqMax").toInt(curFreqMax_);
        if (fmin >= fmax - 50) fmax = fmin + 50;
        curMultiple_ = m; curBins_ = b; curFreqMin_ = fmin; curFreqMax_ = fmax;
        transform_->configure(transform_->sampleRate(), fmin, fmax, b, m);
        if (settingsCallback_) settingsCallback_(m, b, fmin, fmax);
        sendJson(socket, 200, {
            {"status", "ok"},
            {"multiple", m}, {"bins", b}, {"freqMin", fmin}, {"freqMax", fmax},
        });

    } else if (method == "GET" && path == "/api/profiles") {
        QDir dir(profileDir());
        QJsonArray names;
        for (auto& f : dir.entryList({"*.json"}, QDir::Files)) {
            names.append(f.chopped(5)); // strip .json
        }
        sendJsonArray(socket, 200, names);

    } else if (method == "GET" && path.startsWith("/api/profiles/")) {
        QString name = path.mid(QString("/api/profiles/").length()); // after "/api/profiles/"
        QFile file(profileDir() + "/" + name + ".json");
        if (!file.open(QIODevice::ReadOnly)) {
            sendError(socket, 404, "Profile not found: " + name);
        } else {
            auto doc = QJsonDocument::fromJson(file.readAll());
            sendJson(socket, 200, doc.object());
        }

    } else if (method == "POST" && path.startsWith("/api/profiles/")) {
        QString name = path.mid(QString("/api/profiles/").length());
        QDir().mkpath(profileDir());
        QFile file(profileDir() + "/" + name + ".json");
        QJsonObject profile = {
            {"name", name},
            {"multiple", curMultiple_},
            {"bins", curBins_},
            {"freqMin", curFreqMin_},
            {"freqMax", curFreqMax_},
            {"savedAt", QDateTime::currentDateTime().toString(Qt::ISODate)},
        };
        if (!file.open(QIODevice::WriteOnly)) {
            sendError(socket, 500, "Failed to save profile");
        } else {
            file.write(QJsonDocument(profile).toJson(QJsonDocument::Compact));
            sendJson(socket, 201, profile);
        }

    } else if (method == "DELETE" && path.startsWith("/api/profiles/")) {
        QString name = path.mid(QString("/api/profiles/").length());
        QFile file(profileDir() + "/" + name + ".json");
        if (!file.exists()) {
            sendError(socket, 404, "Profile not found: " + name);
        } else {
            file.remove();
            sendOk(socket, "Deleted profile: " + name);
        }

    } else {
        sendError(socket, 404, "Not found: " + path);
    }
}

// ── Response helpers ──

void ApiServer::sendJson(QTcpSocket* socket, int status, const QJsonObject& obj)
{
    QByteArray body = QJsonDocument(obj).toJson(QJsonDocument::Compact);
    QString statusText = (status == 200) ? "OK" : (status == 201) ? "Created" : "Error";
    QByteArray response;
    response += QString("HTTP/1.1 %1 %2\r\n").arg(status).arg(statusText).toUtf8();
    response += "Content-Type: application/json\r\n";
    response += "Access-Control-Allow-Origin: *\r\n";
    response += QString("Content-Length: %1\r\n").arg(body.size()).toUtf8();
    response += "\r\n";
    response += body;
    socket->write(response);
    socket->flush();
    socket->disconnectFromHost();
}

void ApiServer::sendJsonArray(QTcpSocket* socket, int status, const QJsonArray& arr)
{
    QByteArray body = QJsonDocument(arr).toJson(QJsonDocument::Compact);
    QByteArray response;
    response += QString("HTTP/1.1 %1 OK\r\n").arg(status).toUtf8();
    response += "Content-Type: application/json\r\n";
    response += "Access-Control-Allow-Origin: *\r\n";
    response += QString("Content-Length: %1\r\n").arg(body.size()).toUtf8();
    response += "\r\n";
    response += body;
    socket->write(response);
    socket->flush();
    socket->disconnectFromHost();
}

void ApiServer::sendPng(QTcpSocket* socket, const QByteArray& data)
{
    QByteArray response;
    response += "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: image/png\r\n";
    response += "Access-Control-Allow-Origin: *\r\n";
    response += QString("Content-Length: %1\r\n").arg(data.size()).toUtf8();
    response += "\r\n";
    response += data;
    socket->write(response);
    socket->flush();
    socket->disconnectFromHost();
}

void ApiServer::sendError(QTcpSocket* socket, int status, const QString& msg)
{
    sendJson(socket, status, {{"error", msg}});
}

void ApiServer::sendOk(QTcpSocket* socket, const QString& msg)
{
    sendJson(socket, 200, {{"status", "ok"}, {"message", msg}});
}

QString ApiServer::profileDir() const
{
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/profiles";
}
