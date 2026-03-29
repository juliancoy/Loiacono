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

    // MJPEG stream timer — push frames to all connected stream clients
    connect(&streamTimer_, &QTimer::timeout, this, &ApiServer::pushMjpegFrame);

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

    } else if (method == "GET" && path == "/api/stream") {
        startMjpegStream(socket);
        return; // don't close the socket — it stays open for streaming

    } else if (method == "GET" && (path == "/" || path == "/index.html")) {
        // Landing page with live spectrogram and API links
        std::vector<float> spectrum;
        transform_->getSpectrum(spectrum);
        float peak = 0; int peakIdx = 0;
        for (int i = 0; i < (int)spectrum.size(); i++) {
            if (spectrum[i] > peak) { peak = spectrum[i]; peakIdx = i; }
        }
        double peakHz = transform_->numBins() > 0 ? transform_->binFreqHz(peakIdx) : 0;

        QByteArray html = R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Loiacono Spectrogram</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#d0d0e0;font-family:-apple-system,BlinkMacSystemFont,monospace;padding:24px;max-width:960px;margin:0 auto}
h1{color:#a0c0ff;font-weight:400;font-size:1.5rem;margin-bottom:4px}
.subtitle{color:#505070;font-size:.8rem;margin-bottom:20px}
.spectrogram{background:#000;border:1px solid #1a1a2a;border-radius:6px;width:100%;margin-bottom:16px}
.spectrogram img{width:100%;display:block;border-radius:5px;image-rendering:pixelated}
.status{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:20px;padding:12px;background:#0e0e18;border:1px solid #1a1a2a;border-radius:6px}
.stat{display:flex;flex-direction:column}
.stat .label{font-size:.65rem;color:#505070;text-transform:uppercase;letter-spacing:.05em}
.stat .value{font-size:1.1rem;color:#a0c0ff;font-weight:600}
h2{color:#708090;font-size:.9rem;font-weight:400;margin:16px 0 8px;text-transform:uppercase;letter-spacing:.08em}
.endpoints{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.ep{display:flex;gap:8px;padding:8px 12px;background:#0e0e18;border:1px solid #1a1a2a;border-radius:4px;text-decoration:none;transition:border-color .15s}
.ep:hover{border-color:#2a3a5a}
.method{font-size:.7rem;font-weight:700;padding:2px 6px;border-radius:3px;min-width:48px;text-align:center;flex-shrink:0}
.get .method{background:#1a2a1a;color:#60c060}
.put .method{background:#2a2a1a;color:#c0a040}
.post .method{background:#1a1a2a;color:#6080c0}
.delete .method{background:#2a1a1a;color:#c06060}
.ep .path{color:#a0c0ff;font-size:.85rem}
.ep .desc{color:#505070;font-size:.7rem}
.ep-info{display:flex;flex-direction:column}
.refresh-note{color:#404060;font-size:.7rem;text-align:center;margin-top:8px}
</style>
</head>
<body>
<h1>Loiacono Transform</h1>
<p class="subtitle">Rolling spectrogram &mdash; )HTML";
        html += QString("v%1 | Qt %2 | %3")
                    .arg(APP_VERSION).arg(qVersion())
                    .arg(QSysInfo::currentCpuArchitecture()).toUtf8();
        html += R"HTML(</p>
<div class="spectrogram">
<img id="spec" src="/api/stream" alt="Live Spectrogram">
</div>
<div class="status">
<div class="stat"><span class="label">Peak</span><span class="value" id="peak">)HTML";
        html += QString("%1 Hz").arg(peakHz, 0, 'f', 0).toUtf8();
        html += R"HTML(</span></div>
<div class="stat"><span class="label">Bins</span><span class="value" id="bins">)HTML";
        html += QString::number(curBins_).toUtf8();
        html += R"HTML(</span></div>
<div class="stat"><span class="label">Multiple</span><span class="value" id="mult">)HTML";
        html += QString::number(curMultiple_).toUtf8();
        html += R"HTML(</span></div>
<div class="stat"><span class="label">Range</span><span class="value" id="range">)HTML";
        html += QString("%1-%2 Hz").arg(curFreqMin_).arg(curFreqMax_).toUtf8();
        html += R"HTML(</span></div>
<div class="stat"><span class="label">Sample Rate</span><span class="value">48000</span></div>
</div>

<h2>API Endpoints</h2>
<div class="endpoints">
<a class="ep get" href="/api/version"><span class="method">GET</span><div class="ep-info"><span class="path">/api/version</span><span class="desc">App version, platform, architecture</span></div></a>
<a class="ep get" href="/api/status"><span class="method">GET</span><div class="ep-info"><span class="path">/api/status</span><span class="desc">Peak frequency, amplitude, settings</span></div></a>
<a class="ep get" href="/api/screenshot"><span class="method">GET</span><div class="ep-info"><span class="path">/api/screenshot</span><span class="desc">Spectrogram as PNG image</span></div></a>
<a class="ep get" href="/api/stream" target="_blank"><span class="method">GET</span><div class="ep-info"><span class="path">/api/stream</span><span class="desc">Live MJPEG video stream (~30fps)</span></div></a>
<a class="ep get" href="/api/spectrum"><span class="method">GET</span><div class="ep-info"><span class="path">/api/spectrum</span><span class="desc">All frequency bins as JSON</span></div></a>
<a class="ep get" href="/api/profile"><span class="method">GET</span><div class="ep-info"><span class="path">/api/profile</span><span class="desc">Current transform settings</span></div></a>
<a class="ep put" href="#"><span class="method">PUT</span><div class="ep-info"><span class="path">/api/profile</span><span class="desc">Update settings {multiple, bins, freqMin, freqMax}</span></div></a>
<a class="ep get" href="/api/profiles"><span class="method">GET</span><div class="ep-info"><span class="path">/api/profiles</span><span class="desc">List saved profiles</span></div></a>
<a class="ep post" href="#"><span class="method">POST</span><div class="ep-info"><span class="path">/api/profiles/:name</span><span class="desc">Save current settings as profile</span></div></a>
</div>

<script>
setInterval(async()=>{
 try{
  const s=await(await fetch('/api/status')).json();
  document.getElementById('peak').textContent=Math.round(s.peakFrequencyHz)+' Hz';
  document.getElementById('bins').textContent=s.numBins;
  document.getElementById('mult').textContent=s.multiple;
  document.getElementById('range').textContent=s.freqMin+'-'+s.freqMax+' Hz';
 }catch(e){}
},500);
</script>
<p class="refresh-note">Live MJPEG stream at ~30fps | Status refreshes every 500ms</p>
</body>
</html>)HTML";
        sendHtml(socket, html);

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

void ApiServer::sendHtml(QTcpSocket* socket, const QByteArray& html)
{
    QByteArray response;
    response += "HTTP/1.1 200 OK\r\n";
    response += "Content-Type: text/html; charset=utf-8\r\n";
    response += QString("Content-Length: %1\r\n").arg(html.size()).toUtf8();
    response += "\r\n";
    response += html;
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

// ── MJPEG Streaming ──
// The browser <img src="/api/stream"> natively handles multipart JPEG streams.
// No JavaScript needed — the browser decodes each JPEG frame as it arrives.

static constexpr char MJPEG_BOUNDARY[] = "loiaconoframe";

void ApiServer::startMjpegStream(QTcpSocket* socket)
{
    // Send the multipart header — connection stays open
    QByteArray header;
    header += "HTTP/1.1 200 OK\r\n";
    header += "Content-Type: multipart/x-mixed-replace; boundary=";
    header += MJPEG_BOUNDARY;
    header += "\r\n";
    header += "Cache-Control: no-cache, no-store\r\n";
    header += "Access-Control-Allow-Origin: *\r\n";
    header += "Connection: keep-alive\r\n";
    header += "\r\n";
    socket->write(header);
    socket->flush();

    streamClients_.insert(socket);

    // Remove client on disconnect
    connect(socket, &QTcpSocket::disconnected, this, [this, socket]() {
        streamClients_.remove(socket);
        socket->deleteLater();
        if (streamClients_.isEmpty()) {
            streamTimer_.stop();
        }
    });

    // Start the frame timer if not already running (~30 fps)
    if (!streamTimer_.isActive()) {
        streamTimer_.start(33);
    }
}

void ApiServer::pushMjpegFrame()
{
    if (streamClients_.isEmpty()) {
        streamTimer_.stop();
        return;
    }

    // Render the spectrogram to JPEG
    QImage img = spectrogram_->renderToImage();
    QByteArray jpegData;
    QBuffer buf(&jpegData);
    buf.open(QIODevice::WriteOnly);
    img.save(&buf, "JPEG", 80); // quality 80 — good balance of size vs clarity

    // Build the multipart frame
    QByteArray frame;
    frame += "--";
    frame += MJPEG_BOUNDARY;
    frame += "\r\n";
    frame += "Content-Type: image/jpeg\r\n";
    frame += QString("Content-Length: %1\r\n").arg(jpegData.size()).toUtf8();
    frame += "\r\n";
    frame += jpegData;
    frame += "\r\n";

    // Push to all connected clients
    QSet<QTcpSocket*> dead;
    for (auto* client : streamClients_) {
        if (client->state() != QAbstractSocket::ConnectedState) {
            dead.insert(client);
            continue;
        }
        client->write(frame);
        client->flush();
    }
    for (auto* d : dead) {
        streamClients_.remove(d);
        d->deleteLater();
    }
    if (streamClients_.isEmpty()) {
        streamTimer_.stop();
    }
}

QString ApiServer::profileDir() const
{
    return QStandardPaths::writableLocation(QStandardPaths::AppDataLocation) + "/profiles";
}
