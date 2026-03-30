#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QStatusBar>
#include <QGroupBox>
#include <QComboBox>
#include <QSplitter>
#include <QLockFile>
#include <QStandardPaths>
#include <QDir>
#include <QLocalSocket>
#include <QLocalServer>
#include <iostream>

#include "RtAudio.h"
#include "loiacono_rolling.h"
#include "spectrogram_widget.h"
#include "api_server.h"

static constexpr const char* APP_ID = "com.loiacono.spectrogram";
static constexpr quint16 API_PORT_START = 8080;
static constexpr quint16 API_PORT_END = 8090;

// ─── RtAudio callback ───────────────────────────────────────────
static int audioCallback(void*, void* inputBuffer, unsigned int nFrames,
                          double, RtAudioStreamStatus status, void* userData)
{
    if (status) std::cerr << "RtAudio stream overflow\n";
    auto* transform = static_cast<LoiaconoRolling*>(userData);
    transform->processChunk(static_cast<float*>(inputBuffer), static_cast<int>(nFrames));
    return 0;
}

// ─── Single-instance ─────────────────────────────────────────────
static bool ensureSingleInstance(QLockFile& lock, QLocalServer& ipcServer,
                                 QMainWindow* window)
{
    if (lock.tryLock(100)) {
        QLocalServer::removeServer(APP_ID);
        ipcServer.listen(APP_ID);
        QObject::connect(&ipcServer, &QLocalServer::newConnection, [window]() {
            window->raise(); window->activateWindow(); window->showNormal();
        });
        return true;
    }
    QLocalSocket sock;
    sock.connectToServer(APP_ID);
    sock.waitForConnected(500);
    std::cerr << "Another instance is already running.\n";
    return false;
}

// ─── Audio device management ─────────────────────────────────────
static unsigned int currentDeviceId = 0;

static QString openDevice(RtAudio& adc, unsigned int deviceId,
                           double sampleRate, LoiaconoRolling* transform)
{
    if (adc.isStreamOpen()) { adc.stopStream(); adc.closeStream(); }

    auto info = adc.getDeviceInfo(deviceId);
    if (info.inputChannels < 1)
        return QString("Error: '%1' has no input").arg(QString::fromStdString(info.name));

    RtAudio::StreamParameters params;
    params.deviceId = deviceId;
    params.nChannels = 1;
    unsigned int bufferFrames = 256;

    auto err = adc.openStream(nullptr, &params, RTAUDIO_FLOAT32,
                   static_cast<unsigned int>(sampleRate),
                   &bufferFrames, &audioCallback, transform);
    if (err != RTAUDIO_NO_ERROR) return "Error: failed to open stream";

    err = adc.startStream();
    if (err != RTAUDIO_NO_ERROR) return "Error: failed to start stream";

    currentDeviceId = deviceId;
    return QString::fromStdString(info.name);
}

// ─── Slider factory ──────────────────────────────────────────────
static QWidget* makeSlider(const QString& name, QSlider*& slider,
                            int min, int max, int value, const QString& suffix,
                            QLabel*& label)
{
    auto* w = new QWidget;
    auto* lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(4);

    label = new QLabel(QString("%1: %2%3").arg(name).arg(value).arg(suffix));
    label->setFixedWidth(160);
    label->setStyleSheet("color: #b0c0e0; font-size: 11px;");

    slider = new QSlider(Qt::Horizontal);
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setFixedHeight(18);

    lay->addWidget(label);
    lay->addWidget(slider);

    QObject::connect(slider, &QSlider::valueChanged, [label, name, suffix](int v) {
        label->setText(QString("%1: %2%3").arg(name).arg(v).arg(suffix));
    });
    return w;
}

// Float slider (value = slider / scale)
static QWidget* makeFloatSlider(const QString& name, QSlider*& slider, QLabel*& label,
                                 int min, int max, int value, float scale,
                                 const QString& suffix)
{
    auto* w = new QWidget;
    auto* lay = new QHBoxLayout(w);
    lay->setContentsMargins(0, 0, 0, 0);
    lay->setSpacing(4);

    label = new QLabel(QString("%1: %2%3").arg(name).arg(value / scale, 0, 'f', 2).arg(suffix));
    label->setFixedWidth(130);
    label->setStyleSheet("color: #b0c0e0; font-size: 11px;");

    slider = new QSlider(Qt::Horizontal);
    slider->setRange(min, max);
    slider->setValue(value);
    slider->setFixedHeight(18);

    lay->addWidget(label);
    lay->addWidget(slider);

    QObject::connect(slider, &QSlider::valueChanged, [label, name, suffix, scale](int v) {
        label->setText(QString("%1: %2%3").arg(name).arg(v / scale, 0, 'f', 2).arg(suffix));
    });
    return w;
}

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("Loiacono Spectrogram");
    app.setOrganizationName("Loiacono");
    app.setStyle("Fusion");

    // Dark palette
    QPalette pal;
    pal.setColor(QPalette::Window, QColor(18, 18, 28));
    pal.setColor(QPalette::WindowText, QColor(180, 190, 220));
    pal.setColor(QPalette::Base, QColor(14, 14, 22));
    pal.setColor(QPalette::Text, QColor(180, 190, 220));
    pal.setColor(QPalette::Button, QColor(28, 28, 42));
    pal.setColor(QPalette::ButtonText, QColor(160, 180, 220));
    pal.setColor(QPalette::Highlight, QColor(60, 100, 180));
    app.setPalette(pal);

    // Transform
    LoiaconoRolling transform;
    double sampleRate = 48000;
    int freqMin = 100, freqMax = 3000, numBins = 200, multiple = 40;
    transform.configure(sampleRate, freqMin, freqMax, numBins, multiple);

    auto* window = new QMainWindow;
    window->setWindowTitle("Loiacono Transform");
    window->resize(1100, 650);

    // Single instance
    QString lockPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation)
                       + "/loiacono_spectrogram.lock";
    QLockFile lockFile(lockPath);
    QLocalServer ipcServer;
    if (!ensureSingleInstance(lockFile, ipcServer, window)) return 0;

    auto* central = new QWidget;
    auto* mainLayout = new QVBoxLayout(central);
    mainLayout->setContentsMargins(4, 4, 4, 0);
    mainLayout->setSpacing(4);

    // ── Settings row 1: Transform ──
    QSlider *slMultiple, *slBins, *slMin, *slMax;
    QLabel *lbMultiple, *lbBins, *lbMin, *lbMax;

    auto* row1 = new QWidget;
    auto* row1Lay = new QHBoxLayout(row1);
    row1Lay->setContentsMargins(4, 2, 4, 2);
    row1Lay->setSpacing(12);
    row1Lay->addWidget(makeSlider("Multiple", slMultiple, 2, 120, multiple, " periods", lbMultiple));
    row1Lay->addWidget(makeSlider("Bins", slBins, 32, 600, numBins, "", lbBins));
    row1Lay->addWidget(makeSlider("Freq min", slMin, 20, 2000, freqMin, " Hz", lbMin));
    row1Lay->addWidget(makeSlider("Freq max", slMax, 500, 12000, freqMax, " Hz", lbMax));
    mainLayout->addWidget(row1);

    // ── Settings row 2: Gradient + Device ──
    QSlider *slGain, *slGamma, *slFloor;
    QLabel *lbGain, *lbGamma, *lbFloor;

    auto* row2 = new QWidget;
    auto* row2Lay = new QHBoxLayout(row2);
    row2Lay->setContentsMargins(4, 2, 4, 2);
    row2Lay->setSpacing(12);
    row2Lay->addWidget(makeFloatSlider("Gain", slGain, lbGain, 1, 200, 10, 10.0f, "x"));
    row2Lay->addWidget(makeFloatSlider("Gamma", slGamma, lbGamma, 10, 200, 60, 100.0f, ""));
    row2Lay->addWidget(makeFloatSlider("Floor", slFloor, lbFloor, 0, 50, 5, 100.0f, ""));

    // Device selector
    auto* devCombo = new QComboBox;
    devCombo->setMinimumWidth(180);
    devCombo->setStyleSheet("QComboBox { font-size: 11px; }");
    row2Lay->addWidget(new QLabel("Input:"));
    row2Lay->addWidget(devCombo);

    mainLayout->addWidget(row2);

    // ── Spectrogram ──
    auto* spectrogram = new SpectrogramWidget(&transform);
    mainLayout->addWidget(spectrogram, 1);

    window->setCentralWidget(central);
    auto* statusBar = window->statusBar();
    statusBar->setStyleSheet("font-size: 11px; color: #607090;");

    // ── Reconfigure transform ──
    auto reconfigure = [&]() {
        multiple = slMultiple->value();
        numBins = slBins->value();
        freqMin = slMin->value();
        freqMax = slMax->value();
        if (freqMin >= freqMax - 50) freqMax = freqMin + 50;
        transform.configure(sampleRate, freqMin, freqMax, numBins, multiple);
    };
    QObject::connect(slMultiple, &QSlider::valueChanged, reconfigure);
    QObject::connect(slBins, &QSlider::valueChanged, reconfigure);
    QObject::connect(slMin, &QSlider::valueChanged, reconfigure);
    QObject::connect(slMax, &QSlider::valueChanged, reconfigure);

    // ── Gradient controls ──
    QObject::connect(slGain, &QSlider::valueChanged, [spectrogram](int v) {
        spectrogram->setGain(v / 10.0f);
    });
    QObject::connect(slGamma, &QSlider::valueChanged, [spectrogram](int v) {
        spectrogram->setGamma(v / 100.0f);
    });
    QObject::connect(slFloor, &QSlider::valueChanged, [spectrogram](int v) {
        spectrogram->setFloor(v / 100.0f);
    });

    // ── Audio ──
    RtAudio adc;
    auto populateDevices = [&]() {
        devCombo->clear();
        auto ids = adc.getDeviceIds();
        unsigned int defaultIn = adc.getDefaultInputDevice();
        int selectIdx = -1;
        for (auto id : ids) {
            auto info = adc.getDeviceInfo(id);
            if (info.inputChannels < 1) continue;
            QString name = QString::fromStdString(info.name);
            devCombo->addItem(name, id);
            if (id == defaultIn) selectIdx = devCombo->count() - 1;
        }
        if (selectIdx >= 0) devCombo->setCurrentIndex(selectIdx);
    };
    populateDevices();

    auto switchDevice = [&](int comboIdx) {
        if (comboIdx < 0) return;
        unsigned int devId = devCombo->itemData(comboIdx).toUInt();
        QString result = openDevice(adc, devId, sampleRate, &transform);
        statusBar->showMessage(result);
    };
    QObject::connect(devCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), switchDevice);

    // Open default device
    if (adc.getDeviceCount() > 0) {
        QString result = openDevice(adc, adc.getDefaultInputDevice(), sampleRate, &transform);
        statusBar->showMessage(result);
    } else {
        statusBar->showMessage("No audio devices found");
    }

    // ── REST API ──
    auto* api = new ApiServer(&transform, spectrogram, &app);
    api->updateCurrentSettings(multiple, numBins, freqMin, freqMax);

    auto syncApi = [&]() { api->updateCurrentSettings(multiple, numBins, freqMin, freqMax); };
    QObject::connect(slMultiple, &QSlider::valueChanged, syncApi);
    QObject::connect(slBins, &QSlider::valueChanged, syncApi);
    QObject::connect(slMin, &QSlider::valueChanged, syncApi);
    QObject::connect(slMax, &QSlider::valueChanged, syncApi);

    api->setSettingsCallback([&](int m, int b, int fmin, int fmax) {
        slMultiple->setValue(m); slBins->setValue(b);
        slMin->setValue(fmin); slMax->setValue(fmax);
    });

    api->setDeviceListCallback([&adc]() -> QJsonArray {
        QJsonArray arr;
        auto ids = adc.getDeviceIds();
        unsigned int defaultIn = adc.getDefaultInputDevice();
        for (auto id : ids) {
            auto info = adc.getDeviceInfo(id);
            if (info.inputChannels < 1) continue;
            arr.append(QJsonObject{
                {"id", static_cast<int>(id)},
                {"name", QString::fromStdString(info.name)},
                {"channels", static_cast<int>(info.inputChannels)},
                {"sampleRate", static_cast<int>(info.preferredSampleRate)},
                {"isDefault", id == defaultIn},
                {"isActive", id == currentDeviceId},
            });
        }
        return arr;
    });

    api->setDeviceSwitchCallback([&](unsigned int deviceId) -> QString {
        QString result = openDevice(adc, deviceId, sampleRate, &transform);
        statusBar->showMessage(result);
        return result;
    });

    quint16 apiPort = 0;
    for (quint16 port = API_PORT_START; port <= API_PORT_END; port++) {
        if (api->startListening(port)) { apiPort = port; break; }
    }
    if (apiPort) {
        statusBar->showMessage(statusBar->currentMessage() +
                               QString(" | http://localhost:%1").arg(apiPort));
    }

    window->show();
    int ret = app.exec();
    if (adc.isStreamOpen()) adc.closeStream();
    return ret;
}
