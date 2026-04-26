#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QScreen>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QStatusBar>
#include <QGroupBox>
#include <QComboBox>
#include <QCheckBox>
#include <QDialog>
#include <QSplitter>
#include <QTabWidget>
#include <QLockFile>
#include <QStandardPaths>
#include <QDir>
#include <QLocalSocket>
#include <QLocalServer>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QSignalBlocker>
#include <QCommandLineParser>
#include <QTimer>
#include <QElapsedTimer>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>

#include "RtAudio.h"
#include "audio_device_labels.h"
#include "loiacono_rolling.h"
#include "spectrogram_widget.h"
#include "tone_curve_editor.h"
#include "api_server.h"

static constexpr const char* APP_ID = "com.loiacono.spectrogram";
static constexpr quint16 API_PORT_START = 8080;
static constexpr quint16 API_PORT_END = 8090;

enum class SyntheticInputMode {
    Off = 0,
    Sine,
    Sawtooth,
    Square,
};

static SyntheticInputMode syntheticInputModeFromString(const QString& rawMode)
{
    const QString mode = rawMode.trimmed().toLower();
    if (mode == "sine") return SyntheticInputMode::Sine;
    if (mode == "saw" || mode == "sawtooth") return SyntheticInputMode::Sawtooth;
    if (mode == "square") return SyntheticInputMode::Square;
    return SyntheticInputMode::Off;
}

static QString syntheticInputModeName(SyntheticInputMode mode)
{
    switch (mode) {
    case SyntheticInputMode::Sine:
        return "sine";
    case SyntheticInputMode::Sawtooth:
        return "sawtooth";
    case SyntheticInputMode::Square:
        return "square";
    case SyntheticInputMode::Off:
    default:
        return "off";
    }
}

struct SavedUiState {
    int multiple = 40;
    int bins = 200;
    int freqMin = 100;
    int freqMax = 3000;
    int gainTenths = 10;
    int gammaHundredths = 60;
    int floorHundredths = 5;
    int leakinessHundredths = 9995;  // 0.05% leakage (0.9995)
    int displayTenths = 80;
    int modeIndex = 2;
    int deviceId = -1;
    int deviceApi = -1;
    int baseAHundredths = 44000;  // 440.00 Hz default
    int sampleRate = 48000;
    int bufferFrames = 256;
    int bufferCount = 0;
    int audioFlags = 0;
    int temporalWeightingMode = static_cast<int>(LoiaconoRolling::WindowMode::RectangularWindow);
    int normalizationMode = static_cast<int>(LoiaconoRolling::NormalizationMode::Energy);
    int windowLengthMode = static_cast<int>(LoiaconoRolling::WindowLengthMode::PeriodMultiple);
    int algorithmMode = static_cast<int>(LoiaconoRolling::AlgorithmMode::Loiacono);
    int displayNormalizationMode = static_cast<int>(SpectrogramWidget::DisplayNormalizationMode::SmoothedGlobalMax);
    int fixedDisplayReferenceTenths = 10;
    int toneCurveMode = static_cast<int>(SpectrogramWidget::ToneCurveMode::PowerGamma);
    int columnFillMode = static_cast<int>(SpectrogramWidget::ColumnFillMode::DuplicateSnapshot);
    int rollingReconstructionLimit = 24;
    int gridVisible = 1;
    int bufferEdgeMarkersVisible = 0;
    QJsonArray customToneCurve;
    QJsonArray toneCurveEditorGeometry;
};

struct AudioSettings {
    unsigned int sampleRate = 48000;
    unsigned int bufferFrames = 256;
    unsigned int bufferCount = 0;
    RtAudioStreamFlags flags = 0;
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
    state.leakinessHundredths = obj.value("leakinessHundredths").toInt(state.leakinessHundredths);
    state.displayTenths = obj.value("displayTenths").toInt(state.displayTenths);
    state.modeIndex = obj.value("modeIndex").toInt(state.modeIndex);
    state.deviceId = obj.value("deviceId").toInt(state.deviceId);
    state.deviceApi = obj.value("deviceApi").toInt(state.deviceApi);
    state.baseAHundredths = obj.value("baseAHundredths").toInt(state.baseAHundredths);
    state.sampleRate = obj.value("sampleRate").toInt(state.sampleRate);
    state.bufferFrames = obj.value("bufferFrames").toInt(state.bufferFrames);
    state.bufferCount = obj.value("bufferCount").toInt(state.bufferCount);
    state.audioFlags = obj.value("audioFlags").toInt(state.audioFlags);
    state.temporalWeightingMode = obj.value("temporalWeightingMode").toInt(state.temporalWeightingMode);
    state.normalizationMode = obj.value("normalizationMode").toInt(state.normalizationMode);
    state.windowLengthMode = obj.value("windowLengthMode").toInt(state.windowLengthMode);
    state.algorithmMode = obj.value("algorithmMode").toInt(state.algorithmMode);
    state.displayNormalizationMode = obj.value("displayNormalizationMode").toInt(state.displayNormalizationMode);
    state.fixedDisplayReferenceTenths = obj.value("fixedDisplayReferenceTenths").toInt(state.fixedDisplayReferenceTenths);
    state.toneCurveMode = obj.value("toneCurveMode").toInt(state.toneCurveMode);
    state.columnFillMode = obj.value("columnFillMode").toInt(state.columnFillMode);
    state.rollingReconstructionLimit = obj.value("rollingReconstructionLimit").toInt(state.rollingReconstructionLimit);
    state.gridVisible = obj.value("gridVisible").toInt(state.gridVisible);
    state.bufferEdgeMarkersVisible = obj.value("bufferEdgeMarkersVisible").toInt(state.bufferEdgeMarkersVisible);
    if (obj.value("customToneCurve").isArray()) {
        state.customToneCurve = obj.value("customToneCurve").toArray();
    }
    if (obj.value("toneCurveEditorGeometry").isArray()) {
        state.toneCurveEditorGeometry = obj.value("toneCurveEditorGeometry").toArray();
    }
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
        {"leakinessHundredths", state.leakinessHundredths},
        {"displayTenths", state.displayTenths},
        {"modeIndex", state.modeIndex},
        {"deviceId", state.deviceId},
        {"deviceApi", state.deviceApi},
        {"baseAHundredths", state.baseAHundredths},
        {"sampleRate", state.sampleRate},
        {"bufferFrames", state.bufferFrames},
        {"bufferCount", state.bufferCount},
        {"audioFlags", state.audioFlags},
        {"temporalWeightingMode", state.temporalWeightingMode},
        {"normalizationMode", state.normalizationMode},
        {"windowLengthMode", state.windowLengthMode},
        {"algorithmMode", state.algorithmMode},
        {"displayNormalizationMode", state.displayNormalizationMode},
        {"fixedDisplayReferenceTenths", state.fixedDisplayReferenceTenths},
        {"toneCurveMode", state.toneCurveMode},
        {"columnFillMode", state.columnFillMode},
        {"rollingReconstructionLimit", state.rollingReconstructionLimit},
        {"gridVisible", state.gridVisible},
        {"bufferEdgeMarkersVisible", state.bufferEdgeMarkersVisible},
        {"customToneCurve", state.customToneCurve},
        {"toneCurveEditorGeometry", state.toneCurveEditorGeometry},
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
static RtAudio::Api currentDeviceApi = RtAudio::UNSPECIFIED;

struct AudioDeviceChoice {
    RtAudio::Api api = RtAudio::UNSPECIFIED;
    unsigned int id = 0;
    QString backendName;
    QString rawName;
    QString displayName;
    bool isDesktopAudio = false;
    bool isDefault = false;
    unsigned int inputChannels = 0;
    unsigned int outputChannels = 0;
    unsigned int preferredSampleRate = 0;
};

static std::vector<AudioDeviceChoice> enumerateAudioDevices()
{
    std::vector<AudioDeviceChoice> devices;
#if LOIACONO_ENABLE_DESKTOP_AUDIO
    std::vector<RtAudio::Api> apis;
    RtAudio::getCompiledApi(apis);

    for (auto api : apis) {
        if (api == RtAudio::UNSPECIFIED) continue;

        RtAudio backend(api);
        unsigned int defaultIn = backend.getDefaultInputDevice();
        for (auto id : backend.getDeviceIds()) {
            auto info = backend.getDeviceInfo(id);
            if (info.inputChannels < 1) continue;

            QString backendName = QString::fromStdString(RtAudio::getApiDisplayName(api));
            QString rawName = QString::fromStdString(info.name);
            bool isDesktop = isDesktopAudioDevice(backendName, rawName,
                                                  info.inputChannels, info.outputChannels);
            devices.push_back(AudioDeviceChoice{
                api,
                id,
                backendName,
                rawName,
                displayNameForDevice(backendName, rawName,
                                     info.inputChannels, info.outputChannels),
                isDesktop,
                id == defaultIn,
                info.inputChannels,
                info.outputChannels,
                info.preferredSampleRate,
            });
        }
    }

    std::stable_sort(devices.begin(), devices.end(), [](const AudioDeviceChoice& a, const AudioDeviceChoice& b) {
        if (a.isDesktopAudio != b.isDesktopAudio) return a.isDesktopAudio > b.isDesktopAudio;
        if (a.backendName.contains("Pulse", Qt::CaseInsensitive) != b.backendName.contains("Pulse", Qt::CaseInsensitive)) {
            return a.backendName.contains("Pulse", Qt::CaseInsensitive);
        }
        if (a.isDefault != b.isDefault) return a.isDefault > b.isDefault;
        return a.displayName.localeAwareCompare(b.displayName) < 0;
    });
#else
    RtAudio backend;
    unsigned int defaultIn = backend.getDefaultInputDevice();
    for (auto id : backend.getDeviceIds()) {
        auto info = backend.getDeviceInfo(id);
        if (info.inputChannels < 1) continue;

        QString rawName = QString::fromStdString(info.name);
        devices.push_back(AudioDeviceChoice{
            RtAudio::UNSPECIFIED,
            id,
            QString(),
            rawName,
            displayNameForDeviceName(rawName),
            false,
            id == defaultIn,
            info.inputChannels,
            info.outputChannels,
            info.preferredSampleRate,
        });
    }

    std::stable_sort(devices.begin(), devices.end(), [](const AudioDeviceChoice& a, const AudioDeviceChoice& b) {
        if (a.isDefault != b.isDefault) return a.isDefault > b.isDefault;
        return a.displayName.localeAwareCompare(b.displayName) < 0;
    });
#endif
    return devices;
}

static QString openDevice(std::unique_ptr<RtAudio>& adc, RtAudio::Api api, unsigned int deviceId,
                           const AudioSettings& settings,
                           LoiaconoRolling* transform)
{
#if !LOIACONO_ENABLE_DESKTOP_AUDIO
    api = RtAudio::UNSPECIFIED;
#endif
    if (!adc || currentDeviceApi != api) {
        adc = std::make_unique<RtAudio>(api);
    }

    if (adc->isStreamOpen()) {
        adc->stopStream();
        adc->closeStream();
    }

    auto info = adc->getDeviceInfo(deviceId);
    if (info.inputChannels < 1) {
        return QString("Error: '%1' has no input").arg(QString::fromStdString(info.name));
    }

    RtAudio::StreamParameters params;
    params.deviceId = deviceId;
    params.nChannels = 1;
    unsigned int bufferFrames = std::max(16u, settings.bufferFrames);
    RtAudio::StreamOptions options;
    options.flags = settings.flags;
    options.numberOfBuffers = settings.bufferCount;
    options.streamName = "Loiacono Spectrogram";
    if (settings.flags & RTAUDIO_SCHEDULE_REALTIME) {
        options.priority = 10;
    }

    auto err = adc->openStream(nullptr, &params, RTAUDIO_FLOAT32,
                   settings.sampleRate,
                   &bufferFrames, &audioCallback, transform, &options);
    if (err != RTAUDIO_NO_ERROR) return "Error: failed to open stream";

    unsigned int actualSampleRate = adc->getStreamSampleRate();
    if (actualSampleRate == 0) {
        actualSampleRate = settings.sampleRate;
    }
    if (std::abs(transform->sampleRate() - static_cast<double>(actualSampleRate)) > 0.5) {
        auto stats = transform->getStats();
        transform->configure(actualSampleRate,
                             stats.freqMin,
                             stats.freqMax,
                             std::max(1, stats.currentBins),
                             std::max(2, stats.currentMultiple));
    }

    err = adc->startStream();
    if (err != RTAUDIO_NO_ERROR) return "Error: failed to start stream";

    currentDeviceId = deviceId;
    currentDeviceApi = api;
#if LOIACONO_ENABLE_DESKTOP_AUDIO
    QString backendName = QString::fromStdString(RtAudio::getApiDisplayName(api));
    QString sampleRateText = actualSampleRate == settings.sampleRate
        ? QString("%1 Hz").arg(actualSampleRate)
        : QString("%1 Hz requested, %2 Hz actual").arg(settings.sampleRate).arg(actualSampleRate);
    return QString("%1 | %2 | %3 | %4 frames | %5 bufs | %6 frame latency")
        .arg(displayNameForDevice(backendName, QString::fromStdString(info.name),
                                  info.inputChannels, info.outputChannels))
        .arg(backendName)
        .arg(sampleRateText)
        .arg(bufferFrames)
        .arg(options.numberOfBuffers)
        .arg(adc->getStreamLatency());
#else
    QString sampleRateText = actualSampleRate == settings.sampleRate
        ? QString("%1 Hz").arg(actualSampleRate)
        : QString("%1 Hz requested, %2 Hz actual").arg(settings.sampleRate).arg(actualSampleRate);
    return QString("%1 | %2 | %3 frames | %4 bufs | %5 frame latency")
        .arg(displayNameForDeviceName(QString::fromStdString(info.name)))
        .arg(sampleRateText)
        .arg(bufferFrames)
        .arg(options.numberOfBuffers)
        .arg(adc->getStreamLatency());
#endif
}

static QString encodeDeviceKey(RtAudio::Api api, unsigned int id)
{
    return QString("%1:%2").arg(static_cast<int>(api)).arg(id);
}

static bool decodeDeviceKey(const QVariant& data, RtAudio::Api* api, unsigned int* id)
{
    QString key = data.toString();
    QStringList parts = key.split(':');
    if (parts.size() != 2) return false;

    bool apiOk = false;
    bool idOk = false;
    int apiValue = parts[0].toInt(&apiOk);
    unsigned int deviceId = parts[1].toUInt(&idOk);
    if (!apiOk || !idOk) return false;

    *api = static_cast<RtAudio::Api>(apiValue);
    *id = deviceId;
    return true;
}

static QJsonArray rectToJson(const QRect& rect)
{
    return QJsonArray{rect.x(), rect.y(), rect.width(), rect.height()};
}

static QRect rectFromJson(const QJsonArray& arr, const QRect& fallback)
{
    if (arr.size() != 4) return fallback;
    return QRect(arr[0].toInt(fallback.x()),
                 arr[1].toInt(fallback.y()),
                 arr[2].toInt(fallback.width()),
                 arr[3].toInt(fallback.height()));
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
    const QString platformName = QGuiApplication::platformName().toLower();
    const bool gpuDisplayAllowed = !(platformName.contains("offscreen")
                                     || platformName.contains("minimal"));

    QCommandLineParser cmdParser;
    cmdParser.setApplicationDescription("Loiacono rolling spectrogram");
    cmdParser.addHelpOption();
    cmdParser.addVersionOption();
    QCommandLineOption syntheticInputOption(
        "synthetic-input",
        "Synthetic input source: off|sine|sawtooth|square",
        "mode",
        "off");
    QCommandLineOption syntheticFreqOption(
        "synthetic-freq",
        "Synthetic waveform frequency in Hz.",
        "hz",
        "220");
    QCommandLineOption syntheticAmpOption(
        "synthetic-amp",
        "Synthetic waveform amplitude [0.0, 1.0].",
        "amp",
        "0.8");
    cmdParser.addOption(syntheticInputOption);
    cmdParser.addOption(syntheticFreqOption);
    cmdParser.addOption(syntheticAmpOption);
    cmdParser.process(app);

    const QString syntheticModeRaw = cmdParser.value(syntheticInputOption);
    const SyntheticInputMode syntheticMode = syntheticInputModeFromString(syntheticModeRaw);
    const bool syntheticInputEnabled = syntheticMode != SyntheticInputMode::Off;
    bool syntheticFreqOk = false;
    bool syntheticAmpOk = false;
    double syntheticFreqHz = cmdParser.value(syntheticFreqOption).toDouble(&syntheticFreqOk);
    double syntheticAmp = cmdParser.value(syntheticAmpOption).toDouble(&syntheticAmpOk);
    if (!syntheticFreqOk || !std::isfinite(syntheticFreqHz) || syntheticFreqHz <= 0.0) {
        std::cerr << "Invalid --synthetic-freq value: "
                  << cmdParser.value(syntheticFreqOption).toStdString() << "\n";
        return 1;
    }
    if (!syntheticAmpOk || !std::isfinite(syntheticAmp)) {
        std::cerr << "Invalid --synthetic-amp value: "
                  << cmdParser.value(syntheticAmpOption).toStdString() << "\n";
        return 1;
    }
    syntheticAmp = std::clamp(syntheticAmp, 0.0, 1.0);
    if (syntheticModeRaw != "off"
        && syntheticModeRaw != "sine"
        && syntheticModeRaw != "saw"
        && syntheticModeRaw != "sawtooth"
        && syntheticModeRaw != "square") {
        std::cerr << "Invalid --synthetic-input value: "
                  << syntheticModeRaw.toStdString()
                  << " (expected off|sine|sawtooth|square)\n";
        return 1;
    }

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
    SavedUiState savedState = loadSavedUiState();
    if (syntheticInputEnabled) {
        // Deterministic, cleaner visualization preset for isolated synthetic runs.
        savedState.multiple = 96;
        savedState.bins = 1024;
        savedState.freqMin = 40;
        savedState.freqMax = 5000;
        savedState.gainTenths = 12;
        savedState.gammaHundredths = 90;
        savedState.floorHundredths = 10;
        savedState.displayTenths = 80;
        savedState.modeIndex = 2; // Multi-thread + CPU display
        savedState.temporalWeightingMode = static_cast<int>(LoiaconoRolling::WindowMode::HannWindow);
        savedState.normalizationMode = static_cast<int>(LoiaconoRolling::NormalizationMode::Energy);
        savedState.windowLengthMode = static_cast<int>(LoiaconoRolling::WindowLengthMode::PeriodMultiple);
        savedState.algorithmMode = static_cast<int>(LoiaconoRolling::AlgorithmMode::Loiacono);
        savedState.displayNormalizationMode = static_cast<int>(SpectrogramWidget::DisplayNormalizationMode::SmoothedGlobalMax);
        savedState.toneCurveMode = static_cast<int>(SpectrogramWidget::ToneCurveMode::PowerGamma);
        savedState.columnFillMode = static_cast<int>(SpectrogramWidget::ColumnFillMode::DuplicateSnapshot);
        savedState.gridVisible = 1;
        savedState.bufferEdgeMarkersVisible = 0;
    }
    AudioSettings audioSettings;
    audioSettings.sampleRate = static_cast<unsigned int>(std::clamp(savedState.sampleRate, 8000, 192000));
    audioSettings.bufferFrames = static_cast<unsigned int>(std::clamp(savedState.bufferFrames, 16, 4096));
    audioSettings.bufferCount = static_cast<unsigned int>(std::clamp(savedState.bufferCount, 0, 8));
    audioSettings.flags = static_cast<RtAudioStreamFlags>(savedState.audioFlags);
    int freqMin = std::clamp(savedState.freqMin, 20, 2000);
    int freqMax = std::clamp(savedState.freqMax, 500, 12000);
    int numBins = std::clamp(savedState.bins, 32, 2400);
    int multiple = std::clamp(savedState.multiple, 2, 240);
    if (freqMin >= freqMax - 50) freqMax = std::min(12000, freqMin + 50);
    auto temporalWeightingMode = static_cast<LoiaconoRolling::WindowMode>(
        std::clamp(savedState.temporalWeightingMode,
                   static_cast<int>(LoiaconoRolling::WindowMode::RectangularWindow),
                   static_cast<int>(LoiaconoRolling::WindowMode::LeakyWindow)));
    auto normalizationMode = static_cast<LoiaconoRolling::NormalizationMode>(
        std::clamp(savedState.normalizationMode,
                   static_cast<int>(LoiaconoRolling::NormalizationMode::RawSum),
                   static_cast<int>(LoiaconoRolling::NormalizationMode::Energy)));
    auto windowLengthMode = static_cast<LoiaconoRolling::WindowLengthMode>(
        std::clamp(savedState.windowLengthMode,
                   static_cast<int>(LoiaconoRolling::WindowLengthMode::ConstantSamples),
                   static_cast<int>(LoiaconoRolling::WindowLengthMode::PeriodMultiple)));
    auto algorithmMode = static_cast<LoiaconoRolling::AlgorithmMode>(
        std::clamp(savedState.algorithmMode,
                   static_cast<int>(LoiaconoRolling::AlgorithmMode::Loiacono),
                   static_cast<int>(LoiaconoRolling::AlgorithmMode::Goertzel)));
    auto toneCurveMode = static_cast<SpectrogramWidget::ToneCurveMode>(
        std::clamp(savedState.toneCurveMode,
                   static_cast<int>(SpectrogramWidget::ToneCurveMode::PowerGamma),
                   static_cast<int>(SpectrogramWidget::ToneCurveMode::CustomCurve)));
    transform.setWindowMode(temporalWeightingMode);
    transform.setNormalizationMode(normalizationMode);
    transform.setWindowLengthMode(windowLengthMode);
    transform.setAlgorithmMode(algorithmMode);
    transform.configure(audioSettings.sampleRate, freqMin, freqMax, numBins, multiple);

    auto* window = new QMainWindow;
    window->setWindowTitle("Loiacono Transform");
    // Use a conservative size that fits on 1366x768 and larger screens
    // Account for window decorations (~37px title bar + borders)
    window->resize(1024, 650);
    
    // Limit maximum size to prevent window from going off-screen
    QScreen* primaryScreen = QGuiApplication::primaryScreen();
    QRect availableGeometry = primaryScreen->availableGeometry();
    int maxHeight = availableGeometry.height() - 100; // Leave 100px margin for panels
    int maxWidth = availableGeometry.width() - 100;   // Leave 100px margin
    window->setMaximumSize(maxWidth, maxHeight);
    
    // Ensure window is positioned on screen (handle multi-monitor setups)
    QRect frameGeometry = window->frameGeometry();
    int x = qMax(50, (availableGeometry.width() - frameGeometry.width()) / 2);
    int y = qMax(50, (availableGeometry.height() - frameGeometry.height()) / 2);
    window->move(x, y);  // Center on screen with padding

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

    auto* paramsWindow = new QDialog(window);
    paramsWindow->setWindowTitle("Loiacono Parameters");
    paramsWindow->setModal(false);
    paramsWindow->resize(980, 560);
    auto* paramsLayout = new QVBoxLayout(paramsWindow);
    paramsLayout->setContentsMargins(8, 8, 8, 8);
    paramsLayout->setSpacing(8);

    // ── Settings row 1: Transform ──
    QSlider *slMultiple, *slBins, *slMin, *slMax;
    QLabel *lbMultiple, *lbBins, *lbMin, *lbMax;

    auto* row1 = new QGroupBox("Transform");
    auto* row1Lay = new QGridLayout(row1);
    row1Lay->setContentsMargins(8, 10, 8, 8);
    row1Lay->setHorizontalSpacing(12);
    row1Lay->setVerticalSpacing(8);
    auto* fieldWinScale = makeSlider("Win scale", slMultiple, 2, 240, multiple, "", lbMultiple);
    auto* fieldBins = makeSlider("Bins", slBins, 32, 2400, numBins, "", lbBins);
    auto* fieldFreqMin = makeSlider("Freq min", slMin, 20, 2000, freqMin, " Hz", lbMin);
    auto* fieldFreqMax = makeSlider("Freq max", slMax, 500, 12000, freqMax, " Hz", lbMax);

    // ── Settings row 2: Display And Pitch ──
    QSlider *slGain, *slGamma, *slFloor, *slLeakiness, *slDisplaySeconds, *slBaseA, *slDisplayReference;
    QLabel *lbGain, *lbGamma, *lbFloor, *lbLeakiness, *lbDisplaySeconds, *lbBaseA, *lbDisplayReference;

    auto* row2 = new QGroupBox("Display And Pitch");
    auto* row2Lay = new QGridLayout(row2);
    row2Lay->setContentsMargins(8, 10, 8, 8);
    row2Lay->setHorizontalSpacing(12);
    row2Lay->setVerticalSpacing(8);
    row2Lay->addWidget(makeFloatSlider("Gain", slGain, lbGain, 1, 200, std::clamp(savedState.gainTenths, 1, 200), 10.0f, "x"), 0, 0);
    row2Lay->addWidget(makeFloatSlider("Gamma", slGamma, lbGamma, 10, 200, std::clamp(savedState.gammaHundredths, 10, 200), 100.0f, ""), 0, 1);
    row2Lay->addWidget(makeFloatSlider("Floor", slFloor, lbFloor, 0, 50, std::clamp(savedState.floorHundredths, 0, 50), 100.0f, ""), 1, 0);
    row2Lay->addWidget(makeFloatSlider("Leakiness", slLeakiness, lbLeakiness, 9900, 10000, std::clamp(savedState.leakinessHundredths, 9900, 10000), 10000.0f, ""), 1, 1);
    // Fix initial label to show leakage percentage
    double initialLeakage = (10000 - slLeakiness->value()) / 100.0;
    lbLeakiness->setText(QString("Leak: %1%").arg(initialLeakage, 0, 'f', 2));
    row2Lay->addWidget(makeFloatSlider("Displayed time", slDisplaySeconds, lbDisplaySeconds, 10, 300, std::clamp(savedState.displayTenths, 10, 300), 10.0f, " s"), 2, 0);
    row2Lay->addWidget(makeFloatSlider("Disp ref", slDisplayReference, lbDisplayReference, 1, 1000, std::clamp(savedState.fixedDisplayReferenceTenths, 1, 1000), 10.0f, ""), 2, 1);
    // Base A frequency slider (400-500 Hz range)
    int baseAValue = std::clamp(savedState.baseAHundredths, 40000, 50000);
    row2Lay->addWidget(makeFloatSlider("Base A", slBaseA, lbBaseA, 40000, 50000, baseAValue, 100.0f, " Hz"), 3, 0, 1, 2);
    transform.setBaseAFrequency(baseAValue / 100.0);

    // ── Settings row 3: Input + execution/display mode ──
    auto* row3 = new QGroupBox("Audio");
    auto* row3Lay = new QGridLayout(row3);
    row3Lay->setContentsMargins(8, 10, 8, 8);
    row3Lay->setHorizontalSpacing(12);
    row3Lay->setVerticalSpacing(8);

    auto comboStyle = QString("QComboBox { font-size: 11px; min-width: 180px; }");
    auto labelStyle = QString("color: #b0c0e0; font-size: 11px;");

    auto* devCombo = new QComboBox;
    devCombo->setStyleSheet(comboStyle);
    auto* modeCombo = new QComboBox;
    modeCombo->setStyleSheet(comboStyle);
    auto* activeModeLabel = new QLabel;
    activeModeLabel->setStyleSheet("color: #8fb2d9; font-size: 11px;");
    activeModeLabel->setWordWrap(true);
    auto* temporalWeightingCombo = new QComboBox;
    temporalWeightingCombo->setStyleSheet(comboStyle);
    auto* normalizationCombo = new QComboBox;
    normalizationCombo->setStyleSheet(comboStyle);
    auto* windowLengthCombo = new QComboBox;
    windowLengthCombo->setStyleSheet(comboStyle);
    auto* algorithmCombo = new QComboBox;
    algorithmCombo->setStyleSheet(comboStyle);
    auto* toneCurveCombo = new QComboBox;
    toneCurveCombo->setStyleSheet(comboStyle);
    auto* columnFillCombo = new QComboBox;
    columnFillCombo->setStyleSheet(comboStyle);
    auto* rollingReconstructionLimitCombo = new QComboBox;
    rollingReconstructionLimitCombo->setStyleSheet(comboStyle);
    auto* displayNormalizationCombo = new QComboBox;
    displayNormalizationCombo->setStyleSheet(comboStyle);
    auto* sampleRateCombo = new QComboBox;
    sampleRateCombo->setStyleSheet(comboStyle);
    auto* bufferFramesCombo = new QComboBox;
    bufferFramesCombo->setStyleSheet(comboStyle);
    auto* bufferCountCombo = new QComboBox;
    bufferCountCombo->setStyleSheet(comboStyle);
    auto* cbMinLatency = new QCheckBox("Min latency");
    auto* cbRealtime = new QCheckBox("Realtime");
    auto* cbExclusive = new QCheckBox("Exclusive");
    auto* cbAlsaDefault = new QCheckBox("ALSA default");
    auto* cbShowGrid = new QCheckBox("Grid");
    auto* cbBufferEdges = new QCheckBox("Buffer edges");
    for (auto* cb : {cbMinLatency, cbRealtime, cbExclusive, cbAlsaDefault, cbShowGrid, cbBufferEdges}) {
        cb->setStyleSheet("QCheckBox { color: #b0c0e0; font-size: 11px; }");
    }
    cbShowGrid->setChecked(savedState.gridVisible != 0);
    cbBufferEdges->setChecked(savedState.bufferEdgeMarkersVisible != 0);

    const std::vector<unsigned int> sampleRates = {8000, 16000, 22050, 32000, 44100, 48000, 88200, 96000, 192000};
    for (unsigned int rate : sampleRates) {
        sampleRateCombo->addItem(QString::number(rate), static_cast<int>(rate));
    }
    int sampleRateIndex = sampleRateCombo->findData(static_cast<int>(audioSettings.sampleRate));
    sampleRateCombo->setCurrentIndex(sampleRateIndex >= 0 ? sampleRateIndex : sampleRateCombo->findData(48000));

    const std::vector<unsigned int> bufferFrameOptions = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    for (unsigned int frames : bufferFrameOptions) {
        bufferFramesCombo->addItem(QString::number(frames), static_cast<int>(frames));
    }
    int bufferFramesIndex = bufferFramesCombo->findData(static_cast<int>(audioSettings.bufferFrames));
    bufferFramesCombo->setCurrentIndex(bufferFramesIndex >= 0 ? bufferFramesIndex : bufferFramesCombo->findData(256));

    bufferCountCombo->addItem("Auto", 0);
    for (int count = 2; count <= 8; ++count) {
        bufferCountCombo->addItem(QString::number(count), count);
    }
    int bufferCountIndex = bufferCountCombo->findData(static_cast<int>(audioSettings.bufferCount));
    bufferCountCombo->setCurrentIndex(bufferCountIndex >= 0 ? bufferCountIndex : 0);
    cbMinLatency->setChecked(audioSettings.flags & RTAUDIO_MINIMIZE_LATENCY);
    cbRealtime->setChecked(audioSettings.flags & RTAUDIO_SCHEDULE_REALTIME);
    cbExclusive->setChecked(audioSettings.flags & RTAUDIO_HOG_DEVICE);
    cbAlsaDefault->setChecked(audioSettings.flags & RTAUDIO_ALSA_USE_DEFAULT);
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
    addMode("Vulkan compute + CPU display", LoiaconoRolling::ComputeMode::VulkanCompute, false);
    addMode("Vulkan compute + GPU display", LoiaconoRolling::ComputeMode::VulkanCompute, true);
    modeCombo->setCurrentIndex(std::clamp(savedState.modeIndex, 0, modeCombo->count() - 1));
    if (!gpuDisplayAllowed) {
        const int packed = modeCombo->currentData().toInt();
        if ((packed & 1) != 0) {
            modeCombo->setCurrentIndex(std::max(0, modeCombo->currentIndex() - 1));
        }
    }
    temporalWeightingCombo->addItem("Rectangular", static_cast<int>(LoiaconoRolling::WindowMode::RectangularWindow));
    temporalWeightingCombo->addItem("Hann", static_cast<int>(LoiaconoRolling::WindowMode::HannWindow));
    temporalWeightingCombo->addItem("Hamming", static_cast<int>(LoiaconoRolling::WindowMode::HammingWindow));
    temporalWeightingCombo->addItem("Blackman", static_cast<int>(LoiaconoRolling::WindowMode::BlackmanWindow));
    temporalWeightingCombo->addItem("Blackman-Harris", static_cast<int>(LoiaconoRolling::WindowMode::BlackmanHarrisWindow));
    temporalWeightingCombo->addItem("Leaky", static_cast<int>(LoiaconoRolling::WindowMode::LeakyWindow));
    temporalWeightingCombo->setCurrentIndex(
        std::max(0, temporalWeightingCombo->findData(static_cast<int>(temporalWeightingMode))));
    normalizationCombo->addItem("Raw sum", static_cast<int>(LoiaconoRolling::NormalizationMode::RawSum));
    normalizationCombo->addItem("Coherent amplitude", static_cast<int>(LoiaconoRolling::NormalizationMode::CoherentAmplitude));
    normalizationCombo->addItem("Energy", static_cast<int>(LoiaconoRolling::NormalizationMode::Energy));
    normalizationCombo->setCurrentIndex(
        std::max(0, normalizationCombo->findData(static_cast<int>(normalizationMode))));
    windowLengthCombo->addItem("Constant samples", static_cast<int>(LoiaconoRolling::WindowLengthMode::ConstantSamples));
    windowLengthCombo->addItem("Sqrt period", static_cast<int>(LoiaconoRolling::WindowLengthMode::SqrtPeriod));
    windowLengthCombo->addItem("Period multiple", static_cast<int>(LoiaconoRolling::WindowLengthMode::PeriodMultiple));
    windowLengthCombo->setCurrentIndex(
        std::max(0, windowLengthCombo->findData(static_cast<int>(windowLengthMode))));
    algorithmCombo->addItem("Loiacono", static_cast<int>(LoiaconoRolling::AlgorithmMode::Loiacono));
    algorithmCombo->addItem("FFT", static_cast<int>(LoiaconoRolling::AlgorithmMode::FFT));
    algorithmCombo->addItem("Goertzel", static_cast<int>(LoiaconoRolling::AlgorithmMode::Goertzel));
    algorithmCombo->setCurrentIndex(
        std::max(0, algorithmCombo->findData(static_cast<int>(algorithmMode))));
    toneCurveCombo->addItem("Power gamma", static_cast<int>(SpectrogramWidget::ToneCurveMode::PowerGamma));
    toneCurveCombo->addItem("Smoothstep", static_cast<int>(SpectrogramWidget::ToneCurveMode::Smoothstep));
    toneCurveCombo->addItem("Sigmoid", static_cast<int>(SpectrogramWidget::ToneCurveMode::Sigmoid));
    toneCurveCombo->addItem("Custom curve", static_cast<int>(SpectrogramWidget::ToneCurveMode::CustomCurve));
    toneCurveCombo->setCurrentIndex(
        std::max(0, toneCurveCombo->findData(static_cast<int>(toneCurveMode))));
    columnFillCombo->addItem("Duplicate snapshot", static_cast<int>(SpectrogramWidget::ColumnFillMode::DuplicateSnapshot));
    columnFillCombo->addItem("Rolling reconstruction", static_cast<int>(SpectrogramWidget::ColumnFillMode::RollingReconstruction));
    columnFillCombo->setCurrentIndex(
        std::max(0, columnFillCombo->findData(savedState.columnFillMode)));
    for (int limit : {8, 12, 16, 24, 32, 48, 64}) {
        rollingReconstructionLimitCombo->addItem(QString::number(limit), limit);
    }
    rollingReconstructionLimitCombo->setCurrentIndex(
        std::max(0, rollingReconstructionLimitCombo->findData(savedState.rollingReconstructionLimit)));
    displayNormalizationCombo->addItem("Smoothed max", static_cast<int>(SpectrogramWidget::DisplayNormalizationMode::SmoothedGlobalMax));
    displayNormalizationCombo->addItem("Per-frame max", static_cast<int>(SpectrogramWidget::DisplayNormalizationMode::PerFrameMax));
    displayNormalizationCombo->addItem("Peak-hold decay", static_cast<int>(SpectrogramWidget::DisplayNormalizationMode::PeakHoldDecay));
    displayNormalizationCombo->addItem("Fixed reference", static_cast<int>(SpectrogramWidget::DisplayNormalizationMode::FixedReference));
    displayNormalizationCombo->setCurrentIndex(
        std::max(0, displayNormalizationCombo->findData(savedState.displayNormalizationMode)));

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

    row3Lay->addWidget(addLabeledField("Input device", devCombo), 0, 0, 1, 2);
    row3Lay->addWidget(addLabeledField("Execution mode", modeCombo), 1, 0, 1, 2);
    row3Lay->addWidget(activeModeLabel, 2, 0, 1, 2);
    row3Lay->addWidget(addLabeledField("Sample rate", sampleRateCombo), 3, 0);
    row3Lay->addWidget(addLabeledField("Buffer frames", bufferFramesCombo), 3, 1);
    row3Lay->addWidget(addLabeledField("Buffer count", bufferCountCombo), 4, 0);
    auto* audioFlags = new QWidget;
    auto* audioFlagsLay = new QGridLayout(audioFlags);
    audioFlagsLay->setContentsMargins(0, 0, 0, 0);
    audioFlagsLay->setHorizontalSpacing(10);
    audioFlagsLay->setVerticalSpacing(6);
    audioFlagsLay->addWidget(cbMinLatency, 0, 0);
    audioFlagsLay->addWidget(cbRealtime, 0, 1);
    audioFlagsLay->addWidget(cbExclusive, 1, 0);
    audioFlagsLay->addWidget(cbAlsaDefault, 1, 1);
    row3Lay->addWidget(audioFlags, 4, 1);

    auto* row4 = new QGroupBox("Display Options");
    auto* row4Lay = new QGridLayout(row4);
    row4Lay->setContentsMargins(8, 10, 8, 8);
    row4Lay->setHorizontalSpacing(12);
    row4Lay->setVerticalSpacing(8);
    auto* fieldAlgorithm = addLabeledField("Algorithm", algorithmCombo);
    auto* fieldWindowing = addLabeledField("Windowing", temporalWeightingCombo);
    auto* fieldNormalization = addLabeledField("Normalization", normalizationCombo);
    auto* fieldWindowLen = addLabeledField("Window len", windowLengthCombo);
    auto* fieldDisplayNorm = addLabeledField("Display norm", displayNormalizationCombo);
    auto* fieldToneCurve = addLabeledField("Tone curve", toneCurveCombo);
    auto* fieldColumnFill = addLabeledField("Column fill", columnFillCombo);
    auto* fieldReconLimit = addLabeledField("Recon limit", rollingReconstructionLimitCombo);
    row1Lay->addWidget(fieldAlgorithm, 0, 0);
    row1Lay->addWidget(fieldBins, 0, 1);
    row1Lay->addWidget(fieldFreqMin, 1, 0);
    row1Lay->addWidget(fieldFreqMax, 1, 1);
    row1Lay->addWidget(fieldWinScale, 2, 0);
    row1Lay->addWidget(fieldWindowLen, 2, 1);
    row1Lay->addWidget(fieldWindowing, 3, 0);
    row1Lay->addWidget(fieldNormalization, 3, 1);
    row4Lay->addWidget(fieldDisplayNorm, 0, 0);
    row4Lay->addWidget(fieldToneCurve, 0, 1);
    row4Lay->addWidget(fieldColumnFill, 1, 0);
    row4Lay->addWidget(fieldReconLimit, 1, 1);
    auto* toggles = new QWidget;
    auto* togglesLay = new QGridLayout(toggles);
    togglesLay->setContentsMargins(0, 0, 0, 0);
    togglesLay->setHorizontalSpacing(10);
    togglesLay->setVerticalSpacing(6);
    togglesLay->addWidget(cbShowGrid, 0, 0);
    togglesLay->addWidget(cbBufferEdges, 0, 1);
    row4Lay->addWidget(toggles, 2, 0, 1, 2);

    auto* topBar = new QWidget;
    topBar->setMaximumHeight(34);
    auto* topBarLay = new QHBoxLayout(topBar);
    topBarLay->setContentsMargins(4, 2, 4, 2);
    topBarLay->setSpacing(8);
    auto* pauseButton = new QPushButton("Pause");
    pauseButton->setCheckable(true);
    pauseButton->setStyleSheet("QPushButton { font-size: 11px; padding: 4px 10px; }");
    auto* paramsButton = new QPushButton("Parameters...");
    paramsButton->setStyleSheet("QPushButton { font-size: 11px; padding: 4px 10px; }");
    topBarLay->addWidget(pauseButton);
    topBarLay->addWidget(paramsButton);
    topBarLay->addStretch(1);
    mainLayout->addWidget(topBar, 0);

    // ── Spectrogram ──
    auto* spectrogram = new SpectrogramWidget(&transform);
    spectrogram->setGain(slGain->value() / 10.0f);
    spectrogram->setGamma(slGamma->value() / 100.0f);
    spectrogram->setFloor(slFloor->value() / 100.0f);
    spectrogram->setDisplayNormalizationMode(static_cast<SpectrogramWidget::DisplayNormalizationMode>(
        displayNormalizationCombo->currentData().toInt()));
    spectrogram->setFixedDisplayReference(slDisplayReference->value() / 10.0f);
    spectrogram->setToneCurveMode(toneCurveMode);
    spectrogram->setColumnFillMode(static_cast<SpectrogramWidget::ColumnFillMode>(
        columnFillCombo->currentData().toInt()));
    spectrogram->setRollingReconstructionLimit(rollingReconstructionLimitCombo->currentData().toInt());
    spectrogram->setGridVisible(cbShowGrid->isChecked());
    spectrogram->setBufferEdgeMarkersVisible(cbBufferEdges->isChecked());
    spectrogram->setAudioBufferFrames(audioSettings.bufferFrames);
    spectrogram->setCustomToneCurveJson(savedState.customToneCurve);
    transform.setLeakiness(slLeakiness->value() / 10000.0);
    spectrogram->setDisplayedTimeSeconds(slDisplaySeconds->value() / 10.0);
    mainLayout->addWidget(spectrogram, 1);

    auto* toneCurveEditor = new ToneCurveEditorDialog(window);
    toneCurveEditor->setControlPoints(spectrogram->customToneCurve());
    toneCurveEditor->setGeometry(rectFromJson(savedState.toneCurveEditorGeometry, toneCurveEditor->geometry()));
    auto* toneCurveButton = new QPushButton("Curves...");
    toneCurveButton->setStyleSheet("QPushButton { font-size: 11px; padding: 4px 8px; }");
    row4Lay->addWidget(toneCurveButton, 3, 0, 1, 2, Qt::AlignLeft);

    auto* paramsTabs = new QTabWidget(paramsWindow);
    paramsTabs->setDocumentMode(true);

    auto* transformTab = new QWidget(paramsTabs);
    auto* transformTabLayout = new QVBoxLayout(transformTab);
    transformTabLayout->setContentsMargins(6, 6, 6, 6);
    transformTabLayout->setSpacing(8);
    transformTabLayout->addWidget(row1);
    transformTabLayout->addStretch(1);

    auto* displayTab = new QWidget(paramsTabs);
    auto* displayTabLayout = new QVBoxLayout(displayTab);
    displayTabLayout->setContentsMargins(6, 6, 6, 6);
    displayTabLayout->setSpacing(8);
    displayTabLayout->addWidget(row2);
    displayTabLayout->addWidget(row4);
    displayTabLayout->addStretch(1);

    auto* audioTab = new QWidget(paramsTabs);
    auto* audioTabLayout = new QVBoxLayout(audioTab);
    audioTabLayout->setContentsMargins(6, 6, 6, 6);
    audioTabLayout->setSpacing(8);
    audioTabLayout->addWidget(row3);
    audioTabLayout->addStretch(1);

    paramsTabs->addTab(transformTab, "Transform");
    paramsTabs->addTab(displayTab, "Display");
    paramsTabs->addTab(audioTab, "Audio");
    paramsLayout->addWidget(paramsTabs, 1);

    window->setCentralWidget(central);
    auto* statusBar = window->statusBar();
    statusBar->setStyleSheet("font-size: 11px; color: #607090;");

    QObject::connect(pauseButton, &QPushButton::toggled, [spectrogram, pauseButton, statusBar](bool paused) {
        spectrogram->setPaused(paused);
        pauseButton->setText(paused ? "Resume" : "Pause");
        statusBar->showMessage(paused ? "Display paused" : "Display resumed");
    });
    QObject::connect(paramsButton, &QPushButton::clicked, [paramsWindow]() {
        paramsWindow->show();
        paramsWindow->raise();
        paramsWindow->activateWindow();
    });

    auto updateLeakinessLabel = [&]() {
        double leakagePercent = (10000 - slLeakiness->value()) / 100.0;
        bool leakyModeSelected = static_cast<LoiaconoRolling::WindowMode>(
            temporalWeightingCombo->currentData().toInt()) == LoiaconoRolling::WindowMode::LeakyWindow;
        slLeakiness->setEnabled(leakyModeSelected);
        lbLeakiness->setEnabled(leakyModeSelected);
        lbLeakiness->setText(leakyModeSelected
            ? QString("Leak: %1%").arg(leakagePercent, 0, 'f', 2)
            : QString("Leak: off (rect)"));
    };

    auto updateDisplayReferenceUi = [&]() {
        bool fixedMode = static_cast<SpectrogramWidget::DisplayNormalizationMode>(
            displayNormalizationCombo->currentData().toInt()) == SpectrogramWidget::DisplayNormalizationMode::FixedReference;
        slDisplayReference->setEnabled(fixedMode);
        lbDisplayReference->setEnabled(fixedMode);
    };

    auto updateToneCurveUi = [&]() {
        auto mode = static_cast<SpectrogramWidget::ToneCurveMode>(toneCurveCombo->currentData().toInt());
        bool gammaDriven = mode != SpectrogramWidget::ToneCurveMode::CustomCurve;
        slGamma->setEnabled(gammaDriven);
        lbGamma->setEnabled(gammaDriven);
        toneCurveButton->setEnabled(true);
        toneCurveButton->setText(mode == SpectrogramWidget::ToneCurveMode::CustomCurve ? "Edit curve..." : "Curves...");
    };

    auto updateColumnFillUi = [&]() {
        bool rollingMode = static_cast<SpectrogramWidget::ColumnFillMode>(
            columnFillCombo->currentData().toInt()) == SpectrogramWidget::ColumnFillMode::RollingReconstruction;
        rollingReconstructionLimitCombo->setEnabled(rollingMode);
    };
    updateColumnFillUi();

    auto updateTransformUi = [&]() {
        auto mode = static_cast<LoiaconoRolling::AlgorithmMode>(algorithmCombo->currentData().toInt());
        row1->setTitle(QString("Transform: %1").arg(QString::fromLatin1(LoiaconoRolling::algorithmModeName(mode)).toUpper()));
        QString scaleName = mode == LoiaconoRolling::AlgorithmMode::FFT ? "FFT scale" : "Win scale";
        lbMultiple->setText(QString("%1: %2").arg(scaleName).arg(slMultiple->value()));
    };
    updateTransformUi();

    auto updateActiveModeUi = [&]() {
        const int packed = modeCombo->currentData().toInt();
        const auto requestedCompute = static_cast<LoiaconoRolling::ComputeMode>(packed >> 1);
        const bool requestedGpuDisplay = (packed & 1) != 0;
        const auto actualCompute = transform.activeComputeMode();
        const QString displayText = spectrogram->hardwareAccelerationEnabled() ? "gpu-display" : "cpu-display";
        QString text = QString("Active path: %1 + %2")
            .arg(LoiaconoRolling::computeModeName(actualCompute))
            .arg(displayText);
        if (requestedCompute != actualCompute) {
            text += QString("  |  requested %1").arg(LoiaconoRolling::computeModeName(requestedCompute));
        }
        if (requestedGpuDisplay != spectrogram->hardwareAccelerationEnabled()) {
            text += QString("  |  requested %1-display").arg(requestedGpuDisplay ? "gpu" : "cpu");
        }
        activeModeLabel->setText(text);
    };

    auto saveStateNow = [&]() {
        if (syntheticInputEnabled) return;
        SavedUiState current;
        current.multiple = slMultiple->value();
        current.bins = slBins->value();
        current.freqMin = slMin->value();
        current.freqMax = slMax->value();
        current.gainTenths = slGain->value();
        current.gammaHundredths = slGamma->value();
        current.floorHundredths = slFloor->value();
        current.leakinessHundredths = slLeakiness->value();
        current.displayTenths = static_cast<int>(std::lround(spectrogram->displayedTimeSeconds() * 10.0));
        current.baseAHundredths = slBaseA->value();
        current.modeIndex = modeCombo->currentIndex();
        current.deviceId = static_cast<int>(currentDeviceId);
        current.deviceApi = static_cast<int>(currentDeviceApi);
        if (devCombo->currentIndex() >= 0) {
            RtAudio::Api api = RtAudio::UNSPECIFIED;
            unsigned int deviceId = 0;
            if (decodeDeviceKey(devCombo->currentData(), &api, &deviceId)) {
                current.deviceId = static_cast<int>(deviceId);
                current.deviceApi = static_cast<int>(api);
            }
        }
        current.sampleRate = static_cast<int>(audioSettings.sampleRate);
        current.bufferFrames = static_cast<int>(audioSettings.bufferFrames);
        current.bufferCount = static_cast<int>(audioSettings.bufferCount);
        current.audioFlags = static_cast<int>(audioSettings.flags);
        current.temporalWeightingMode = temporalWeightingCombo->currentData().toInt();
        current.normalizationMode = normalizationCombo->currentData().toInt();
        current.windowLengthMode = windowLengthCombo->currentData().toInt();
        current.algorithmMode = algorithmCombo->currentData().toInt();
        current.displayNormalizationMode = displayNormalizationCombo->currentData().toInt();
        current.fixedDisplayReferenceTenths = slDisplayReference->value();
        current.toneCurveMode = toneCurveCombo->currentData().toInt();
        current.columnFillMode = columnFillCombo->currentData().toInt();
        current.rollingReconstructionLimit = rollingReconstructionLimitCombo->currentData().toInt();
        current.gridVisible = cbShowGrid->isChecked() ? 1 : 0;
        current.bufferEdgeMarkersVisible = cbBufferEdges->isChecked() ? 1 : 0;
        current.customToneCurve = spectrogram->customToneCurveJson();
        current.toneCurveEditorGeometry = rectToJson(toneCurveEditor->geometry());
        saveUiState(current);
    };

    // ── Reconfigure transform ──
    auto reconfigure = [&]() {
        multiple = slMultiple->value();
        numBins = slBins->value();
        freqMin = slMin->value();
        freqMax = slMax->value();
        if (freqMin >= freqMax - 50) freqMax = freqMin + 50;
        transform.setWindowMode(static_cast<LoiaconoRolling::WindowMode>(
            temporalWeightingCombo->currentData().toInt()));
        transform.setNormalizationMode(static_cast<LoiaconoRolling::NormalizationMode>(
            normalizationCombo->currentData().toInt()));
        transform.setWindowLengthMode(static_cast<LoiaconoRolling::WindowLengthMode>(
            windowLengthCombo->currentData().toInt()));
        transform.setAlgorithmMode(static_cast<LoiaconoRolling::AlgorithmMode>(
            algorithmCombo->currentData().toInt()));
        transform.configure(audioSettings.sampleRate, freqMin, freqMax, numBins, multiple);
        spectrogram->setAudioBufferFrames(audioSettings.bufferFrames);
        spectrogram->setDisplayNormalizationMode(static_cast<SpectrogramWidget::DisplayNormalizationMode>(
            displayNormalizationCombo->currentData().toInt()));
        spectrogram->setFixedDisplayReference(slDisplayReference->value() / 10.0f);
        spectrogram->setToneCurveMode(static_cast<SpectrogramWidget::ToneCurveMode>(
            toneCurveCombo->currentData().toInt()));
        spectrogram->setColumnFillMode(static_cast<SpectrogramWidget::ColumnFillMode>(
            columnFillCombo->currentData().toInt()));
        spectrogram->setRollingReconstructionLimit(rollingReconstructionLimitCombo->currentData().toInt());
        spectrogram->resetHistory();
        saveStateNow();
    };
    QObject::connect(slMultiple, &QSlider::valueChanged, reconfigure);
    QObject::connect(slMultiple, &QSlider::valueChanged, [&](int) { updateTransformUi(); });
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
    QObject::connect(slDisplayReference, &QSlider::valueChanged, [spectrogram, saveStateNow](int v) {
        spectrogram->setFixedDisplayReference(v / 10.0f);
        saveStateNow();
    });
    QObject::connect(slLeakiness, &QSlider::valueChanged, [&transform, lbLeakiness, saveStateNow](int v) {
        transform.setLeakiness(v / 10000.0);
        saveStateNow();
    });
    QObject::connect(slDisplaySeconds, &QSlider::valueChanged, [spectrogram](int v) {
        spectrogram->setDisplayedTimeSeconds(v / 10.0);
    });
    QObject::connect(slBaseA, &QSlider::valueChanged, [&transform, lbBaseA, saveStateNow](int v) {
        double freq = v / 100.0;
        transform.setBaseAFrequency(freq);
        lbBaseA->setText(QString("Base A: %1 Hz").arg(freq, 0, 'f', 2));
        saveStateNow();
    });
    QObject::connect(spectrogram, &SpectrogramWidget::displayedTimeChanged, [slDisplaySeconds, saveStateNow](double seconds) {
        int sliderValue = static_cast<int>(std::lround(seconds * 10.0));
        sliderValue = std::clamp(sliderValue, slDisplaySeconds->minimum(), slDisplaySeconds->maximum());
        if (slDisplaySeconds->value() != sliderValue) {
            QSignalBlocker blocker(slDisplaySeconds);
            slDisplaySeconds->setValue(sliderValue);
        }
        saveStateNow();
    });
    spectrogram->setDisplayedTimeSeconds(slDisplaySeconds->value() / 10.0);
    QObject::connect(spectrogram, &SpectrogramWidget::frequencyRangeChanged, [slMin, slMax](int newMin, int newMax) {
        slMin->setValue(std::clamp(newMin, slMin->minimum(), slMin->maximum()));
        slMax->setValue(std::clamp(newMax, slMax->minimum(), slMax->maximum()));
    });

    QObject::connect(modeCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&transform, spectrogram, statusBar, modeCombo, saveStateNow, updateActiveModeUi, gpuDisplayAllowed](int index) {
        int packed = modeCombo->itemData(index).toInt();
        auto mode = static_cast<LoiaconoRolling::ComputeMode>(packed >> 1);
        bool gpuDisplay = (packed & 1) != 0;
        if (!gpuDisplayAllowed) {
            gpuDisplay = false;
        }
        transform.setComputeMode(mode);
        spectrogram->setHardwareAccelerationEnabled(gpuDisplay);

        QString message = QString("Mode: %1").arg(modeCombo->itemText(index));
        if (mode == LoiaconoRolling::ComputeMode::GpuCompute && !transform.gpuComputeAvailable()) {
            message += " | GPU compute unavailable, using multi-thread CPU";
        } else if (mode == LoiaconoRolling::ComputeMode::VulkanCompute && !transform.vulkanComputeAvailable()) {
            message += " | Vulkan compute unavailable, using multi-thread CPU";
        }
        statusBar->showMessage(message);
        updateActiveModeUi();
        saveStateNow();
    });
    // Explicitly trigger the signal to set initial mode
    emit modeCombo->currentIndexChanged(modeCombo->currentIndex());
    updateActiveModeUi();

    // ── Audio ──
    std::unique_ptr<RtAudio> adc;
    QTimer syntheticTimer;
    QElapsedTimer syntheticElapsed;
    std::vector<float> syntheticChunk;
    syntheticChunk.resize(std::max(16u, audioSettings.bufferFrames), 0.0f);
    uint64_t syntheticSampleIndex = 0;
    qint64 syntheticLastNs = 0;
    double syntheticPendingSamples = 0.0;

    constexpr double kTwoPi = 6.28318530717958647692;
    auto pushSyntheticChunk = [&]() {
        if (!syntheticInputEnabled) return;
        if (!syntheticElapsed.isValid()) {
            syntheticElapsed.start();
            syntheticLastNs = syntheticElapsed.nsecsElapsed();
        }
        const qint64 nowNs = syntheticElapsed.nsecsElapsed();
        const qint64 deltaNs = std::max<qint64>(0, nowNs - syntheticLastNs);
        syntheticLastNs = nowNs;
        syntheticPendingSamples += (static_cast<double>(deltaNs) * static_cast<double>(audioSettings.sampleRate)) / 1.0e9;
        int frames = static_cast<int>(std::floor(syntheticPendingSamples));
        if (frames < 1) frames = 1;
        syntheticPendingSamples -= frames;

        if (static_cast<int>(syntheticChunk.size()) != frames) {
            syntheticChunk.resize(frames, 0.0f);
        }
        const double sampleRate = std::max(1.0, static_cast<double>(audioSettings.sampleRate));
        for (int i = 0; i < frames; ++i) {
            const double phase = std::fmod((static_cast<double>(syntheticSampleIndex++) * syntheticFreqHz) / sampleRate, 1.0);
            double sample = 0.0;
            switch (syntheticMode) {
            case SyntheticInputMode::Sine:
                sample = std::sin(phase * kTwoPi);
                break;
            case SyntheticInputMode::Sawtooth:
                sample = (2.0 * phase) - 1.0;
                break;
            case SyntheticInputMode::Square:
                sample = (phase < 0.5) ? 1.0 : -1.0;
                break;
            case SyntheticInputMode::Off:
            default:
                sample = 0.0;
                break;
            }
            syntheticChunk[i] = static_cast<float>(sample * syntheticAmp);
        }
        transform.processChunk(syntheticChunk.data(), frames);
    };
    auto restartSyntheticTimer = [&]() {
        if (!syntheticInputEnabled) return;
        syntheticTimer.stop();
        syntheticPendingSamples = 0.0;
        syntheticElapsed.restart();
        syntheticLastNs = syntheticElapsed.nsecsElapsed();
        syntheticTimer.start(5);
    };
    QObject::connect(&syntheticTimer, &QTimer::timeout, [&]() { pushSyntheticChunk(); });

    auto populateDevices = [&]() {
        devCombo->clear();
        int selectIdx = -1;
        auto devices = enumerateAudioDevices();
        for (const auto& device : devices) {
            devCombo->addItem(device.displayName, encodeDeviceKey(device.api, device.id));
            devCombo->setItemData(devCombo->count() - 1, device.rawName, Qt::UserRole + 1);
            devCombo->setItemData(devCombo->count() - 1, device.backendName, Qt::UserRole + 2);
            if (static_cast<int>(device.id) == savedState.deviceId
                && (savedState.deviceApi < 0 || static_cast<int>(device.api) == savedState.deviceApi)) {
                selectIdx = devCombo->count() - 1;
            } else if (selectIdx < 0 && device.isDesktopAudio) {
                selectIdx = devCombo->count() - 1;
            } else if (selectIdx < 0 && device.isDefault) {
                selectIdx = devCombo->count() - 1;
            }
        }
        if (selectIdx >= 0) devCombo->setCurrentIndex(selectIdx);
    };
    populateDevices();

    auto applyAudioSettings = [&]() {
        audioSettings.sampleRate = static_cast<unsigned int>(sampleRateCombo->currentData().toInt());
        audioSettings.bufferFrames = static_cast<unsigned int>(bufferFramesCombo->currentData().toInt());
        audioSettings.bufferCount = static_cast<unsigned int>(bufferCountCombo->currentData().toInt());
        int flags = 0;
        if (cbMinLatency->isChecked()) flags |= RTAUDIO_MINIMIZE_LATENCY;
        if (cbRealtime->isChecked()) flags |= RTAUDIO_SCHEDULE_REALTIME;
        if (cbExclusive->isChecked()) flags |= RTAUDIO_HOG_DEVICE;
        if (cbAlsaDefault->isChecked()) flags |= RTAUDIO_ALSA_USE_DEFAULT;
        audioSettings.flags = static_cast<RtAudioStreamFlags>(flags);

        transform.setWindowMode(static_cast<LoiaconoRolling::WindowMode>(
            temporalWeightingCombo->currentData().toInt()));
        transform.setNormalizationMode(static_cast<LoiaconoRolling::NormalizationMode>(
            normalizationCombo->currentData().toInt()));
        transform.setWindowLengthMode(static_cast<LoiaconoRolling::WindowLengthMode>(
            windowLengthCombo->currentData().toInt()));
        transform.setAlgorithmMode(static_cast<LoiaconoRolling::AlgorithmMode>(
            algorithmCombo->currentData().toInt()));
        transform.configure(audioSettings.sampleRate, freqMin, freqMax, numBins, multiple);
        spectrogram->setAudioBufferFrames(audioSettings.bufferFrames);
        spectrogram->resetHistory();
        if (syntheticInputEnabled) {
            restartSyntheticTimer();
            statusBar->showMessage(QString("Synthetic %1: %2 Hz @ %3 amp")
                                       .arg(syntheticInputModeName(syntheticMode))
                                       .arg(syntheticFreqHz, 0, 'f', 2)
                                       .arg(syntheticAmp, 0, 'f', 2));
        } else if (devCombo->currentIndex() >= 0) {
            RtAudio::Api api = RtAudio::UNSPECIFIED;
            unsigned int devId = 0;
            if (decodeDeviceKey(devCombo->currentData(), &api, &devId)) {
                QString result = openDevice(adc, api, devId, audioSettings, &transform);
                statusBar->showMessage(result);
            }
        }
        updateLeakinessLabel();
        updateActiveModeUi();
        saveStateNow();
    };

    auto switchDevice = [&](int comboIdx) {
        if (syntheticInputEnabled) return;
        if (comboIdx < 0) return;
        RtAudio::Api api = RtAudio::UNSPECIFIED;
        unsigned int devId = 0;
        if (!decodeDeviceKey(devCombo->itemData(comboIdx), &api, &devId)) return;
        QString result = openDevice(adc, api, devId, audioSettings, &transform);
        statusBar->showMessage(result);
        saveStateNow();
    };
    QObject::connect(devCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), switchDevice);
    QObject::connect(sampleRateCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) { applyAudioSettings(); });
    QObject::connect(bufferFramesCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) { applyAudioSettings(); });
    QObject::connect(bufferCountCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) { applyAudioSettings(); });
    QObject::connect(temporalWeightingCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        updateLeakinessLabel();
        reconfigure();
        saveStateNow();
    });
    QObject::connect(normalizationCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        reconfigure();
        saveStateNow();
    });
    QObject::connect(windowLengthCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        reconfigure();
        saveStateNow();
    });
    QObject::connect(algorithmCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        updateTransformUi();
        reconfigure();
        emit modeCombo->currentIndexChanged(modeCombo->currentIndex());
        updateActiveModeUi();
        saveStateNow();
    });
    QObject::connect(toneCurveCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        spectrogram->setToneCurveMode(static_cast<SpectrogramWidget::ToneCurveMode>(
            toneCurveCombo->currentData().toInt()));
        updateToneCurveUi();
        spectrogram->resetHistory();
        saveStateNow();
    });
    QObject::connect(columnFillCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        spectrogram->setColumnFillMode(static_cast<SpectrogramWidget::ColumnFillMode>(
            columnFillCombo->currentData().toInt()));
        updateColumnFillUi();
        spectrogram->resetHistory();
        saveStateNow();
    });
    QObject::connect(rollingReconstructionLimitCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        spectrogram->setRollingReconstructionLimit(rollingReconstructionLimitCombo->currentData().toInt());
        if (spectrogram->columnFillMode() == SpectrogramWidget::ColumnFillMode::RollingReconstruction) {
            spectrogram->resetHistory();
        }
        saveStateNow();
    });
    QObject::connect(cbShowGrid, &QCheckBox::toggled, [&](bool checked) {
        spectrogram->setGridVisible(checked);
        saveStateNow();
    });
    QObject::connect(cbBufferEdges, &QCheckBox::toggled, [&](bool checked) {
        spectrogram->setBufferEdgeMarkersVisible(checked);
        saveStateNow();
    });
    QObject::connect(displayNormalizationCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), [&](int) {
        spectrogram->setDisplayNormalizationMode(static_cast<SpectrogramWidget::DisplayNormalizationMode>(
            displayNormalizationCombo->currentData().toInt()));
        updateDisplayReferenceUi();
        spectrogram->resetHistory();
        saveStateNow();
    });
    QObject::connect(toneCurveButton, &QPushButton::clicked, [=]() {
        toneCurveEditor->setControlPoints(spectrogram->customToneCurve());
        toneCurveEditor->show();
        toneCurveEditor->raise();
        toneCurveEditor->activateWindow();
    });
    QObject::connect(toneCurveEditor, &ToneCurveEditorDialog::curveChanged, [&](const std::vector<QPointF>& points) {
        spectrogram->setCustomToneCurve(points);
        if (spectrogram->toneCurveMode() == SpectrogramWidget::ToneCurveMode::CustomCurve) {
            spectrogram->resetHistory();
        }
        saveStateNow();
    });
    QObject::connect(cbMinLatency, &QCheckBox::toggled, [&](bool) { applyAudioSettings(); });
    QObject::connect(cbRealtime, &QCheckBox::toggled, [&](bool) { applyAudioSettings(); });
    QObject::connect(cbExclusive, &QCheckBox::toggled, [&](bool) { applyAudioSettings(); });
    QObject::connect(cbAlsaDefault, &QCheckBox::toggled, [&](bool) { applyAudioSettings(); });
    QObject::connect(slLeakiness, &QSlider::valueChanged, [&](int) { updateLeakinessLabel(); });
    updateLeakinessLabel();
    updateDisplayReferenceUi();
    updateToneCurveUi();

    // Open default device
    if (syntheticInputEnabled) {
        restartSyntheticTimer();
        statusBar->showMessage(QString("Synthetic %1: %2 Hz @ %3 amp")
                                   .arg(syntheticInputModeName(syntheticMode))
                                   .arg(syntheticFreqHz, 0, 'f', 2)
                                   .arg(syntheticAmp, 0, 'f', 2));
    } else if (devCombo->currentIndex() >= 0) {
        RtAudio::Api api = RtAudio::UNSPECIFIED;
        unsigned int devId = 0;
        if (decodeDeviceKey(devCombo->currentData(), &api, &devId)) {
            QString result = openDevice(adc, api, devId, audioSettings, &transform);
            statusBar->showMessage(result);
        }
    } else {
        statusBar->showMessage("No audio devices found");
    }

    // ── REST API ──
    auto* api = new ApiServer(&transform, spectrogram, &app);
    api->updateCurrentSettings(multiple, numBins, freqMin, freqMax, transform.baseAFrequency());

    auto syncApi = [&]() { api->updateCurrentSettings(multiple, numBins, freqMin, freqMax, transform.baseAFrequency()); };
    QObject::connect(slMultiple, &QSlider::valueChanged, syncApi);
    QObject::connect(slBins, &QSlider::valueChanged, syncApi);
    QObject::connect(slMin, &QSlider::valueChanged, syncApi);
    QObject::connect(slMax, &QSlider::valueChanged, syncApi);

    api->setSettingsCallback([&](int m, int b, int fmin, int fmax) {
        slMultiple->setValue(m); slBins->setValue(b);
        slMin->setValue(fmin); slMax->setValue(fmax);
    });

    api->setDeviceListCallback([]() -> QJsonArray {
        QJsonArray arr;
        for (const auto& device : enumerateAudioDevices()) {
            arr.append(QJsonObject{
                {"id", encodeDeviceKey(device.api, device.id)},
                {"deviceId", static_cast<int>(device.id)},
                {"api", static_cast<int>(device.api)},
                {"apiName", device.backendName},
                {"name", device.displayName},
                {"rawName", device.rawName},
                {"channels", static_cast<int>(device.inputChannels)},
                {"sampleRate", static_cast<int>(device.preferredSampleRate)},
                {"isDefault", device.isDefault},
                {"isDesktopAudio", device.isDesktopAudio},
                {"isActive", device.id == currentDeviceId && device.api == currentDeviceApi},
            });
        }
        return arr;
    });

    api->setDeviceSwitchCallback([&](const QString& deviceKey) -> QString {
        RtAudio::Api api = RtAudio::UNSPECIFIED;
        unsigned int deviceId = 0;
        if (!decodeDeviceKey(deviceKey, &api, &deviceId)) {
            return "Error: invalid device id";
        }
        QString result = openDevice(adc, api, deviceId, audioSettings, &transform);
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
    paramsWindow->move(window->frameGeometry().right() + 12, window->frameGeometry().top());
    paramsWindow->show();
    int ret = app.exec();
    saveStateNow();
    syntheticTimer.stop();
    if (adc && adc->isStreamOpen()) adc->closeStream();
    return ret;
}
