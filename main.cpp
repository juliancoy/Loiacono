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
#include <QCheckBox>
#include <QSplitter>
#include <QLockFile>
#include <QStandardPaths>
#include <QDir>
#include <QLocalSocket>
#include <QLocalServer>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QSignalBlocker>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "RtAudio.h"
#include "loiacono_rolling.h"
#include "spectrogram_widget.h"
#include "api_server.h"

static constexpr const char* APP_ID = "com.loiacono.spectrogram";
static constexpr quint16 API_PORT_START = 8080;
static constexpr quint16 API_PORT_END = 8090;

struct SavedUiState {
    int multiple = 40;
    int bins = 200;
    int freqMin = 100;
    int freqMax = 3000;
    int gainTenths = 10;
    int gammaHundredths = 60;
    int floorHundredths = 5;
    int displayTenths = 80;
    int modeIndex = 2;
    int deviceId = -1;
};

static QString settingsFilePath()
{
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
    QDir().mkpath(dir);
    return QDir(dir).filePath("settings.json");
}

static SavedUiState loadSavedUiState()
{
    SavedUiState state;
    QFile file(settingsFilePath());
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) return state;

    const auto doc = QJsonDocument::fromJson(file.readAll());
    if (!doc.isObject()) return state;
    const auto obj = doc.object();
    state.multiple = obj.value("multiple").toInt(state.multiple);
    state.bins = obj.value("bins").toInt(state.bins);
    state.freqMin = obj.value("freqMin").toInt(state.freqMin);
    state.freqMax = obj.value("freqMax").toInt(state.freqMax);
    state.gainTenths = obj.value("gainTenths").toInt(state.gainTenths);
    state.gammaHundredths = obj.value("gammaHundredths").toInt(state.gammaHundredths);
    state.floorHundredths = obj.value("floorHundredths").toInt(state.floorHundredths);
    state.displayTenths = obj.value("displayTenths").toInt(state.displayTenths);
    state.modeIndex = obj.value("modeIndex").toInt(state.modeIndex);
    state.deviceId = obj.value("deviceId").toInt(state.deviceId);
    return state;
}

static void saveUiState(const SavedUiState& state)
{
    const QJsonObject obj{
        {"multiple", state.multiple},
        {"bins", state.bins},
        {"freqMin", state.freqMin},
        {"freqMax", state.freqMax},
        {"gainTenths", state.gainTenths},
        {"gammaHundredths", state.gammaHundredths},
        {"floorHundredths", state.floorHundredths},
        {"displayTenths", state.displayTenths},
        {"modeIndex", state.modeIndex},
        {"deviceId", state.deviceId},
    };

    QFile file(settingsFilePath());
    if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate | QIODevice::Text)) return;
    file.write(QJsonDocument(obj).toJson(QJsonDocument::Indented));
}

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
    SavedUiState savedState = loadSavedUiState();
    int freqMin = std::clamp(savedState.freqMin, 20, 2000);
    int freqMax = std::clamp(savedState.freqMax, 500, 12000);
    int numBins = std::clamp(savedState.bins, 32, 600);
    int multiple = std::clamp(savedState.multiple, 2, 120);
    if (freqMin >= freqMax - 50) freqMax = std::min(12000, freqMin + 50);
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

    // ── Settings row 2: Visual tuning ──
    QSlider *slGain, *slGamma, *slFloor, *slDisplaySeconds;
    QLabel *lbGain, *lbGamma, *lbFloor, *lbDisplaySeconds;

    auto* row2 = new QWidget;
    auto* row2Lay = new QHBoxLayout(row2);
    row2Lay->setContentsMargins(4, 2, 4, 2);
    row2Lay->setSpacing(12);
    row2Lay->addWidget(makeFloatSlider("Gain", slGain, lbGain, 1, 200, std::clamp(savedState.gainTenths, 1, 200), 10.0f, "x"));
    row2Lay->addWidget(makeFloatSlider("Gamma", slGamma, lbGamma, 10, 200, std::clamp(savedState.gammaHundredths, 10, 200), 100.0f, ""));
    row2Lay->addWidget(makeFloatSlider("Floor", slFloor, lbFloor, 0, 50, std::clamp(savedState.floorHundredths, 0, 50), 100.0f, ""));
    row2Lay->addWidget(makeFloatSlider("Displayed time", slDisplaySeconds, lbDisplaySeconds, 10, 300, std::clamp(savedState.displayTenths, 10, 300), 10.0f, " s"));
    mainLayout->addWidget(row2);

    // ── Settings row 3: Input + execution/display mode ──
    auto* row3 = new QWidget;
    auto* row3Lay = new QHBoxLayout(row3);
    row3Lay->setContentsMargins(4, 2, 4, 2);
    row3Lay->setSpacing(12);

    auto comboStyle = QString("QComboBox { font-size: 11px; min-width: 180px; }");
    auto labelStyle = QString("color: #b0c0e0; font-size: 11px;");

    auto* devCombo = new QComboBox;
    devCombo->setStyleSheet(comboStyle);
    auto* modeCombo = new QComboBox;
    modeCombo->setStyleSheet(comboStyle);
    auto addMode = [&](const QString& label, LoiaconoRolling::ComputeMode computeMode, bool gpuDisplay) {
        int packed = (static_cast<int>(computeMode) << 1) | (gpuDisplay ? 1 : 0);
        modeCombo->addItem(label, packed);
    };
    addMode("Single-thread + CPU display", LoiaconoRolling::ComputeMode::SingleThread, false);
    addMode("Single-thread + GPU display", LoiaconoRolling::ComputeMode::SingleThread, true);
    addMode("Multi-thread + CPU display", LoiaconoRolling::ComputeMode::MultiThread, false);
    addMode("Multi-thread + GPU display", LoiaconoRolling::ComputeMode::MultiThread, true);
    addMode("GPU compute + CPU display", LoiaconoRolling::ComputeMode::GpuCompute, false);
    addMode("GPU compute + GPU display", LoiaconoRolling::ComputeMode::GpuCompute, true);
    modeCombo->setCurrentIndex(std::clamp(savedState.modeIndex, 0, modeCombo->count() - 1));

    auto addLabeledField = [&](const QString& labelText, QWidget* field) {
        auto* box = new QWidget;
        auto* lay = new QVBoxLayout(box);
        lay->setContentsMargins(0, 0, 0, 0);
        lay->setSpacing(2);
        auto* label = new QLabel(labelText);
        label->setStyleSheet(labelStyle);
        lay->addWidget(label);
        lay->addWidget(field);
        return box;
    };

    row3Lay->addWidget(addLabeledField("Input device", devCombo), 1);
    row3Lay->addWidget(addLabeledField("Execution mode", modeCombo));
    mainLayout->addWidget(row3);

    // ── Spectrogram ──
    auto* spectrogram = new SpectrogramWidget(&transform);
    spectrogram->setGain(slGain->value() / 10.0f);
    spectrogram->setGamma(slGamma->value() / 100.0f);
    spectrogram->setFloor(slFloor->value() / 100.0f);
    spectrogram->setDisplayedTimeSeconds(slDisplaySeconds->value() / 10.0);
    mainLayout->addWidget(spectrogram, 1);

    window->setCentralWidget(central);
    auto* statusBar = window->statusBar();
    statusBar->setStyleSheet("font-size: 11px; color: #607090;");

    auto saveStateNow = [&]() {
        SavedUiState current;
        current.multiple = slMultiple->value();
        current.bins = slBins->value();
        current.freqMin = slMin->value();
        current.freqMax = slMax->value();
        current.gainTenths = slGain->value();
        current.gammaHundredths = slGamma->value();
        current.floorHundredths = slFloor->value();
        current.displayTenths = slDisplaySeconds->value();
        current.modeIndex = modeCombo->currentIndex();
        current.deviceId = devCombo->currentIndex() >= 0 ? devCombo->currentData().toInt() : currentDeviceId;
        saveUiState(current);
    };

    // ── Reconfigure transform ──
    auto reconfigure = [&]() {
        multiple = slMultiple->value();
        numBins = slBins->value();
        freqMin = slMin->value();
        freqMax = slMax->value();
        if (freqMin >= freqMax - 50) freqMax = freqMin + 50;
        transform.configure(sampleRate, freqMin, freqMax, numBins, multiple);
        spectrogram->resetHistory();
        saveStateNow();
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
    QObject::connect(slDisplaySeconds, &QSlider::valueChanged, [spectrogram](int v) {
        spectrogram->setDisplayedTimeSeconds(v / 10.0);
    });
    spectrogram->setDisplayedTimeSeconds(slDisplaySeconds->value() / 10.0);
    QObject::connect(spectrogram, &SpectrogramWidget::displayedTimeChanged, [slDisplaySeconds](double seconds) {
        int sliderValue = static_cast<int>(std::lround(seconds * 10.0));
        sliderValue = std::clamp(sliderValue, slDisplaySeconds->minimum(), slDisplaySeconds->maximum());
        slDisplaySeconds->setValue(sliderValue);
    });
    QObject::connect(spectrogram, &SpectrogramWidget::frequencyRangeChanged, [slMin, slMax](int newMin, int newMax) {
        slMin->setValue(std::clamp(newMin, slMin->minimum(), slMin->maximum()));
        slMax->setValue(std::clamp(newMax, slMax->minimum(), slMax->maximum()));
    });

    QObject::connect(modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&transform, spectrogram, statusBar, modeCombo, saveStateNow](int index) {
        int packed = modeCombo->itemData(index).toInt();
        auto mode = static_cast<LoiaconoRolling::ComputeMode>(packed >> 1);
        bool gpuDisplay = (packed & 1) != 0;
        transform.setComputeMode(mode);
        spectrogram->setHardwareAccelerationEnabled(gpuDisplay);

        QString message = QString("Mode: %1").arg(modeCombo->itemText(index));
        if (mode == LoiaconoRolling::ComputeMode::GpuCompute && !transform.gpuComputeAvailable()) {
            message += " | GPU compute unavailable, using multi-thread CPU";
        }
        statusBar->showMessage(message);
        saveStateNow();
    });
    modeCombo->setCurrentIndex(modeCombo->currentIndex());

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
            if (static_cast<int>(id) == savedState.deviceId) selectIdx = devCombo->count() - 1;
            else if (selectIdx < 0 && id == defaultIn) selectIdx = devCombo->count() - 1;
        }
        if (selectIdx >= 0) devCombo->setCurrentIndex(selectIdx);
    };
    populateDevices();

    auto switchDevice = [&](int comboIdx) {
        if (comboIdx < 0) return;
        unsigned int devId = devCombo->itemData(comboIdx).toUInt();
        QString result = openDevice(adc, devId, sampleRate, &transform);
        statusBar->showMessage(result);
        saveStateNow();
    };
    QObject::connect(devCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), switchDevice);

    // Open default device
    if (adc.getDeviceCount() > 0 && devCombo->currentIndex() >= 0) {
        QString result = openDevice(adc, devCombo->currentData().toUInt(), sampleRate, &transform);
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

    saveStateNow();
    window->show();
    int ret = app.exec();
    saveStateNow();
    if (adc.isStreamOpen()) adc.closeStream();
    return ret;
}
