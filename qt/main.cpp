#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QStatusBar>
#include <QGroupBox>
#include <QLockFile>
#include <QStandardPaths>
#include <QDir>
#include <QLocalSocket>
#include <QLocalServer>
#include <iostream>
#include <csignal>

#include "RtAudio.h"
#include "loiacono_rolling.h"
#include "spectrogram_widget.h"
#include "api_server.h"

static constexpr const char* APP_ID = "com.loiacono.spectrogram";
static constexpr quint16 API_PORT_START = 8080;
static constexpr quint16 API_PORT_END = 8090;

// ─── RtAudio callback ───────────────────────────────────────────
static int audioCallback(void* /*outputBuffer*/, void* inputBuffer,
                          unsigned int nFrames, double /*streamTime*/,
                          RtAudioStreamStatus status, void* userData)
{
    if (status) std::cerr << "RtAudio stream overflow\n";
    auto* transform = static_cast<LoiaconoRolling*>(userData);
    auto* in = static_cast<float*>(inputBuffer);
    transform->processChunk(in, static_cast<int>(nFrames));
    return 0;
}

// ─── Labeled slider helper ───────────────────────────────────────
struct LabeledSlider {
    QLabel* label;
    QSlider* slider;
    QWidget* widget;

    static LabeledSlider create(const QString& name, int min, int max, int value,
                                 const QString& suffix = "")
    {
        auto* w = new QWidget;
        auto* layout = new QVBoxLayout(w);
        layout->setContentsMargins(0, 0, 0, 0);
        layout->setSpacing(2);

        auto* lbl = new QLabel(QString("%1: %2%3").arg(name).arg(value).arg(suffix));
        auto* sl = new QSlider(Qt::Horizontal);
        sl->setRange(min, max);
        sl->setValue(value);

        QObject::connect(sl, &QSlider::valueChanged, [lbl, name, suffix](int v) {
            lbl->setText(QString("%1: %2%3").arg(name).arg(v).arg(suffix));
        });

        return {lbl, sl, w};
    }
};

// ─── Single-instance enforcement ─────────────────────────────────
// Returns true if we are the only instance. If another is running,
// signals it to raise its window and returns false.
static bool ensureSingleInstance(QLockFile& lock, QLocalServer& ipcServer,
                                 QMainWindow* window)
{
    if (lock.tryLock(100)) {
        // We got the lock — we're the primary instance.
        // Listen for "raise" signals from future launches.
        QLocalServer::removeServer(APP_ID);
        ipcServer.listen(APP_ID);
        QObject::connect(&ipcServer, &QLocalServer::newConnection, [window]() {
            window->raise();
            window->activateWindow();
            window->showNormal();
        });
        return true;
    }

    // Another instance holds the lock — tell it to raise and exit.
    QLocalSocket sock;
    sock.connectToServer(APP_ID);
    if (sock.waitForConnected(500)) {
        sock.disconnectFromServer();
    }
    std::cerr << "Another instance is already running. Bringing it to front.\n";
    return false;
}

// ─── Audio setup (non-fatal) ─────────────────────────────────────
static QString setupAudio(RtAudio& adc, double sampleRate,
                           LoiaconoRolling* transform)
{
    if (adc.getDeviceCount() < 1) {
        return "No audio devices found — spectrogram will be blank until a mic is connected";
    }

    RtAudio::StreamParameters params;
    params.deviceId = adc.getDefaultInputDevice();
    params.nChannels = 1;
    unsigned int bufferFrames = 256;

    auto err = adc.openStream(nullptr, &params, RTAUDIO_FLOAT32,
                   static_cast<unsigned int>(sampleRate),
                   &bufferFrames, &audioCallback, transform);
    if (err != RTAUDIO_NO_ERROR) {
        return "Failed to open audio stream — check microphone permissions";
    }

    err = adc.startStream();
    if (err != RTAUDIO_NO_ERROR) {
        return "Failed to start audio stream";
    }

    auto info = adc.getDeviceInfo(params.deviceId);
    return QString("Listening on: %1 | %2 Hz | buffer: %3")
        .arg(QString::fromStdString(info.name))
        .arg(sampleRate)
        .arg(bufferFrames);
}

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("Loiacono Spectrogram");
    app.setOrganizationName("Loiacono");

    // Transform
    LoiaconoRolling transform;
    double sampleRate = 48000;
    int freqMin = 100, freqMax = 3000, numBins = 200, multiple = 40;
    transform.configure(sampleRate, freqMin, freqMax, numBins, multiple);

    // Main window (created early so single-instance can raise it)
    auto* window = new QMainWindow;
    window->setWindowTitle("Loiacono Transform \u2014 Rolling Spectrogram");
    window->resize(900, 550);

    // ── Single instance check ──
    QString lockPath = QStandardPaths::writableLocation(QStandardPaths::TempLocation)
                       + "/loiacono_spectrogram.lock";
    QLockFile lockFile(lockPath);
    QLocalServer ipcServer;
    if (!ensureSingleInstance(lockFile, ipcServer, window)) {
        return 0; // other instance raised, we exit cleanly
    }

    auto* central = new QWidget;
    auto* mainLayout = new QVBoxLayout(central);

    // ── Settings panel ──
    auto* settingsBox = new QGroupBox("Settings");
    auto* settingsLayout = new QHBoxLayout(settingsBox);

    auto temporalSlider = LabeledSlider::create("Temporal precision", 2, 120, multiple, " periods");
    auto binsSlider = LabeledSlider::create("Frequency precision", 32, 600, numBins, " bins");
    auto minSlider = LabeledSlider::create("Freq min", 20, 2000, freqMin, " Hz");
    auto maxSlider = LabeledSlider::create("Freq max", 500, 12000, freqMax, " Hz");

    settingsLayout->addWidget(temporalSlider.widget);
    settingsLayout->addWidget(binsSlider.widget);
    settingsLayout->addWidget(minSlider.widget);
    settingsLayout->addWidget(maxSlider.widget);

    mainLayout->addWidget(settingsBox);

    // ── Spectrogram ──
    auto* spectrogram = new SpectrogramWidget(&transform);
    mainLayout->addWidget(spectrogram, 1);

    window->setCentralWidget(central);

    auto* statusBar = window->statusBar();
    statusBar->showMessage("Starting...");

    // ── Settings handlers ──
    auto reconfigure = [&]() {
        multiple = temporalSlider.slider->value();
        numBins = binsSlider.slider->value();
        freqMin = minSlider.slider->value();
        freqMax = maxSlider.slider->value();
        if (freqMin >= freqMax - 50) freqMax = freqMin + 50;
        transform.configure(sampleRate, freqMin, freqMax, numBins, multiple);
    };
    QObject::connect(temporalSlider.slider, &QSlider::valueChanged, reconfigure);
    QObject::connect(binsSlider.slider, &QSlider::valueChanged, reconfigure);
    QObject::connect(minSlider.slider, &QSlider::valueChanged, reconfigure);
    QObject::connect(maxSlider.slider, &QSlider::valueChanged, reconfigure);

    // ── Audio (non-fatal) ──
    RtAudio adc;
    QString audioStatus = setupAudio(adc, sampleRate, &transform);
    statusBar->showMessage(audioStatus);

    // ── REST API server (try ports 8080-8090) ──
    auto* api = new ApiServer(&transform, spectrogram, &app);
    api->updateCurrentSettings(multiple, numBins, freqMin, freqMax);

    auto syncApi = [&]() {
        api->updateCurrentSettings(multiple, numBins, freqMin, freqMax);
    };
    QObject::connect(temporalSlider.slider, &QSlider::valueChanged, syncApi);
    QObject::connect(binsSlider.slider, &QSlider::valueChanged, syncApi);
    QObject::connect(minSlider.slider, &QSlider::valueChanged, syncApi);
    QObject::connect(maxSlider.slider, &QSlider::valueChanged, syncApi);

    api->setSettingsCallback([&](int m, int b, int fmin, int fmax) {
        temporalSlider.slider->setValue(m);
        binsSlider.slider->setValue(b);
        minSlider.slider->setValue(fmin);
        maxSlider.slider->setValue(fmax);
    });

    quint16 apiPort = 0;
    for (quint16 port = API_PORT_START; port <= API_PORT_END; port++) {
        if (api->startListening(port)) {
            apiPort = port;
            break;
        }
    }
    if (apiPort) {
        statusBar->showMessage(statusBar->currentMessage() +
                               QString(" | API: http://localhost:%1").arg(apiPort));
    } else {
        statusBar->showMessage(statusBar->currentMessage() +
                               " | API: failed to bind (ports 8080-8090 busy)");
    }

    window->show();
    int ret = app.exec();

    // Cleanup
    if (adc.isStreamOpen()) adc.closeStream();
    return ret;
}
