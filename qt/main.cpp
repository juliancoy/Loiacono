#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSlider>
#include <QLabel>
#include <QPushButton>
#include <QStatusBar>
#include <QGroupBox>
#include <iostream>

#include "RtAudio.h"
#include "loiacono_rolling.h"
#include "spectrogram_widget.h"
#include "api_server.h"

// ─── RtAudio callback ───────────────────────────────────────────
// Called on the audio thread with each buffer of samples.
// Feeds directly into the rolling Loiacono transform.
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

        layout->addWidget(lbl);
        layout->addWidget(sl);

        QObject::connect(sl, &QSlider::valueChanged, [lbl, name, suffix](int v) {
            lbl->setText(QString("%1: %2%3").arg(name).arg(v).arg(suffix));
        });

        return {lbl, sl, w};
    }
};

// ─── Main ────────────────────────────────────────────────────────
int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    app.setApplicationName("Loiacono Spectrogram");

    // Transform
    LoiaconoRolling transform;
    double sampleRate = 48000;
    int freqMin = 100, freqMax = 3000, numBins = 200, multiple = 40;
    transform.configure(sampleRate, freqMin, freqMax, numBins, multiple);

    // Main window
    auto* window = new QMainWindow;
    window->setWindowTitle("Loiacono Transform — Rolling Spectrogram");
    window->resize(900, 550);

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

    // Status bar
    auto* statusBar = window->statusBar();
    statusBar->showMessage("Starting audio...");

    // ── Settings handlers — reconfigure transform on change ──
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

    // ── RtAudio setup ──
    RtAudio adc;
    if (adc.getDeviceCount() < 1) {
        statusBar->showMessage("No audio devices found!");
    } else {
        RtAudio::StreamParameters params;
        params.deviceId = adc.getDefaultInputDevice();
        params.nChannels = 1;

        unsigned int bufferFrames = 256;
        auto err = adc.openStream(nullptr, &params, RTAUDIO_FLOAT32,
                       static_cast<unsigned int>(sampleRate),
                       &bufferFrames, &audioCallback, &transform);
        if (err != RTAUDIO_NO_ERROR) {
            statusBar->showMessage("Failed to open audio stream");
        } else {
            err = adc.startStream();
            if (err != RTAUDIO_NO_ERROR) {
                statusBar->showMessage("Failed to start audio stream");
            } else {
                auto info = adc.getDeviceInfo(params.deviceId);
                statusBar->showMessage(
                    QString("Listening on: %1 | %2 Hz | buffer: %3")
                        .arg(QString::fromStdString(info.name))
                        .arg(sampleRate)
                        .arg(bufferFrames));
            }
        }
    }

    // ── REST API server ──
    auto* api = new ApiServer(&transform, spectrogram, &app);
    api->updateCurrentSettings(multiple, numBins, freqMin, freqMax);

    // When sliders change, sync to API server
    auto syncApi = [&]() {
        api->updateCurrentSettings(multiple, numBins, freqMin, freqMax);
    };
    QObject::connect(temporalSlider.slider, &QSlider::valueChanged, syncApi);
    QObject::connect(binsSlider.slider, &QSlider::valueChanged, syncApi);
    QObject::connect(minSlider.slider, &QSlider::valueChanged, syncApi);
    QObject::connect(maxSlider.slider, &QSlider::valueChanged, syncApi);

    // When API changes settings, sync sliders back
    api->setSettingsCallback([&](int m, int b, int fmin, int fmax) {
        temporalSlider.slider->setValue(m);
        binsSlider.slider->setValue(b);
        minSlider.slider->setValue(fmin);
        maxSlider.slider->setValue(fmax);
    });

    quint16 apiPort = 8080;
    if (api->startListening(apiPort)) {
        statusBar->showMessage(statusBar->currentMessage() +
                               QString(" | API: http://localhost:%1").arg(apiPort));
    }

    window->show();
    int ret = app.exec();

    // Cleanup
    if (adc.isStreamOpen()) adc.closeStream();
    return ret;
}
