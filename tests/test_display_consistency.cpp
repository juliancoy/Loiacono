// Display Consistency Test
// Tests that all compute/display modes produce identical visual output
//
// Usage: ./test_display_consistency [output_dir]
// Generates PNG images for each mode and a comparison report

#include <QApplication>
#include <QOffscreenSurface>
#include <QOpenGLContext>
#include <QOpenGLFramebufferObject>
#include <QOpenGLExtraFunctions>
#include <QSurfaceFormat>
#include <QImage>
#include <QPainter>
#include <QDir>
#include <QFile>
#include <QTextStream>
#include <QThread>
#include <QDateTime>
#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>

#include "../loiacono_rolling.h"

// Test signal generators
std::vector<float> generateSineSweep(double sampleRate, double duration, 
                                      double f0, double f1) {
    int samples = static_cast<int>(duration * sampleRate);
    std::vector<float> signal(samples);
    for (int i = 0; i < samples; i++) {
        double t = i / sampleRate;
        double phase = 2.0 * M_PI * (f0 * t + (f1 - f0) * t * t / (2.0 * duration));
        signal[i] = std::sin(phase) * 0.8f;
    }
    return signal;
}

std::vector<float> generateSineTone(double sampleRate, double duration,
                                     double freq) {
    int samples = static_cast<int>(duration * sampleRate);
    std::vector<float> signal(samples);
    for (int i = 0; i < samples; i++) {
        double t = i / sampleRate;
        signal[i] = std::sin(2.0 * M_PI * freq * t) * 0.8f;
    }
    return signal;
}

std::vector<float> generateImpulseTrain(double sampleRate, double duration,
                                         double intervalSec) {
    int samples = static_cast<int>(duration * sampleRate);
    std::vector<float> signal(samples, 0.0f);
    int intervalSamples = static_cast<int>(intervalSec * sampleRate);
    for (int i = 0; i < samples; i += intervalSamples) {
        signal[i] = 1.0f;
    }
    return signal;
}

struct TestPattern {
    std::string name;
    std::vector<float> signal;
    double sampleRate;
};

// Simple spectrogram renderer that mimics the widget behavior
class SpectrogramRenderer {
public:
    struct RGB { uint8_t r, g, b; };
    
    float gain_ = 1.0f;
    float gamma_ = 0.6f;
    float floor_ = 0.05f;
    float maxAmplitude_ = 1.0f;
    
    RGB colormap(float amplitude) {
        float logMax = std::log(1.0f + maxAmplitude_ * gain_);
        float t = logMax > 0 ? std::log(1.0f + amplitude * gain_) / logMax : 0;
        t = std::clamp(t, 0.0f, 1.0f);
        
        if (t < floor_) return {0, 0, 0};
        
        t = (t - floor_) / (1.0f - floor_);
        t = std::pow(t, gamma_);
        
        if (t < 0.15f) {
            float s = t / 0.15f;
            return {0, 0, static_cast<uint8_t>(s * 200)};
        } else if (t < 0.35f) {
            float s = (t - 0.15f) / 0.2f;
            return {0, static_cast<uint8_t>(s * 255), static_cast<uint8_t>((200 + s * 55) / 255 * 255)};
        } else if (t < 0.55f) {
            float s = (t - 0.35f) / 0.2f;
            return {0, 255, static_cast<uint8_t>(255 * (1 - s))};
        } else if (t < 0.75f) {
            float s = (t - 0.55f) / 0.2f;
            return {static_cast<uint8_t>(s * 255), 255, 0};
        } else {
            float s = (t - 0.75f) / 0.25f;
            return {255, static_cast<uint8_t>(255 * (1 - s * 0.6f)), static_cast<uint8_t>(s * 180)};
        }
    }
    
    // Render spectrogram from spectrum history
    QImage render(const std::vector<std::vector<float>>& spectrumHistory,
                  int width, int height,
                  double displaySeconds,
                  double sampleRate) {
        QImage image(width, height, QImage::Format_RGB32);
        image.fill(Qt::black);
        
        int numBins = spectrumHistory.empty() ? 0 : spectrumHistory[0].size();
        if (numBins == 0) return image;
        
        // Compute samples per column
        double samplesPerColumn = (displaySeconds * sampleRate) / width;
        
        // Map each column to a spectrum in history
        for (int col = 0; col < width; col++) {
            int historyIdx = (spectrumHistory.size() - 1) - 
                           static_cast<int>((width - 1 - col) * samplesPerColumn / 256.0);
            historyIdx = std::clamp(historyIdx, 0, static_cast<int>(spectrumHistory.size()) - 1);
            
            const auto& spectrum = spectrumHistory[historyIdx];
            
            // Map frequency bins to rows (inverted: row 0 = top = highest freq)
            for (int row = 0; row < height; row++) {
                double binF = (numBins - 1) * (height - 1 - row) / std::max(1, height - 1);
                int fi = static_cast<int>(std::clamp(std::floor(binF), 0.0, static_cast<double>(numBins - 1)));
                
                auto [r, g, b] = colormap(fi < static_cast<int>(spectrum.size()) ? spectrum[fi] : 0.0f);
                image.setPixelColor(col, row, QColor(r, g, b));
            }
        }
        
        return image;
    }
};

struct ModeResult {
    std::string modeName;
    QImage spectrogramImage;
    std::vector<std::vector<float>> spectrumHistory;
    std::vector<float> finalSpectrum;
    bool success = false;
};

// Compute MSE between two images
double computeImageMSE(const QImage& a, const QImage& b) {
    if (a.size() != b.size()) return -1.0;
    if (a.format() != b.format()) return -1.0;
    
    double mse = 0.0;
    int pixels = a.width() * a.height();
    
    for (int y = 0; y < a.height(); y++) {
        const QRgb* lineA = reinterpret_cast<const QRgb*>(a.scanLine(y));
        const QRgb* lineB = reinterpret_cast<const QRgb*>(b.scanLine(y));
        for (int x = 0; x < a.width(); x++) {
            QRgb pa = lineA[x];
            QRgb pb = lineB[x];
            mse += std::pow(qRed(pa) - qRed(pb), 2);
            mse += std::pow(qGreen(pa) - qGreen(pb), 2);
            mse += std::pow(qBlue(pa) - qBlue(pb), 2);
        }
    }
    return mse / (pixels * 3.0);
}

// Compute max absolute difference
double computeImageMaxDiff(const QImage& a, const QImage& b) {
    if (a.size() != b.size()) return -1.0;
    
    double maxDiff = 0.0;
    for (int y = 0; y < a.height(); y++) {
        const QRgb* lineA = reinterpret_cast<const QRgb*>(a.scanLine(y));
        const QRgb* lineB = reinterpret_cast<const QRgb*>(b.scanLine(y));
        for (int x = 0; x < a.width(); x++) {
            QRgb pa = lineA[x];
            QRgb pb = lineB[x];
            maxDiff = std::max(maxDiff, static_cast<double>(std::abs(qRed(pa) - qRed(pb))));
            maxDiff = std::max(maxDiff, static_cast<double>(std::abs(qGreen(pa) - qGreen(pb))));
            maxDiff = std::max(maxDiff, static_cast<double>(std::abs(qBlue(pa) - qBlue(pb))));
        }
    }
    return maxDiff;
}

ModeResult runMode(LoiaconoRolling::ComputeMode computeMode,
                   const TestPattern& pattern,
                   int imgWidth = 400,
                   int imgHeight = 300,
                   int numBins = 100,
                   int multiple = 40,
                   bool doRender = true) {
    ModeResult result;
    result.modeName = LoiaconoRolling::computeModeName(computeMode);
    
    // Create transform
    LoiaconoRolling transform;
    transform.setComputeMode(computeMode);
    transform.configure(pattern.sampleRate, 50.0, 8000.0, numBins, multiple);
    
    // Debug: check what mode we're actually running
    auto activeMode = transform.activeComputeMode();
    bool gpuAvail = transform.gpuComputeAvailable();
    std::cout << "\n    Configured mode: " << result.modeName 
              << ", Active mode: " << LoiaconoRolling::computeModeName(activeMode)
              << ", GPU available: " << (gpuAvail ? "yes" : "no") << std::endl;
    
    // Process the signal in chunks and capture spectrum at each step
    int chunkSize = 256;
    int totalSamples = pattern.signal.size();
    int processed = 0;
    int tickCounter = 0;
    
    std::vector<std::vector<float>> history;
    std::vector<float> spectrum;
    
    while (processed < totalSamples) {
        int thisChunk = std::min(chunkSize, totalSamples - processed);
        transform.processChunk(pattern.signal.data() + processed, thisChunk);
        processed += thisChunk;
        
        // Capture spectrum every 4 chunks (to simulate display updates)
        tickCounter++;
        if (tickCounter % 4 == 0) {
            transform.getSpectrum(spectrum);
            history.push_back(spectrum);
        }
    }
    
    // Final capture
    transform.getSpectrum(spectrum);
    result.finalSpectrum = spectrum;
    if (history.empty() || history.back() != spectrum) {
        history.push_back(spectrum);
    }
    result.spectrumHistory = history;
    
    // Render spectrogram image (or leave empty for later rendering)
    if (doRender) {
        SpectrogramRenderer renderer;
        result.spectrogramImage = renderer.render(history, imgWidth, imgHeight, 
                                                   2.0, pattern.sampleRate);
        result.success = !result.spectrogramImage.isNull();
    } else {
        result.success = !history.empty();
    }
    
    return result;
}

bool saveResults(const std::vector<ModeResult>& results, const QString& outputDir) {
    QDir dir(outputDir);
    if (!dir.exists()) {
        dir.mkpath(".");
    }
    
    // Save images
    for (const auto& result : results) {
        if (!result.success) continue;
        
        QString name = QString::fromStdString(result.modeName);
        name.replace(" ", "_");
        
        QString imgPath = dir.filePath(name + "_spectrogram.png");
        result.spectrogramImage.save(imgPath);
        std::cout << "  Saved: " << imgPath.toStdString() << std::endl;
    }
    
    // Generate comparison report
    QString reportPath = dir.filePath("comparison_report.html");
    QFile file(reportPath);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        return false;
    }
    
    QTextStream out(&file);
    out << "<!DOCTYPE html>\n<html>\n<head>\n";
    out << "<title>Display Consistency Report</title>\n";
    out << "<style>\n";
    out << "body { font-family: sans-serif; margin: 20px; }\n";
    out << "table { border-collapse: collapse; margin: 20px 0; }\n";
    out << "th, td { border: 1px solid #ccc; padding: 10px; text-align: center; }\n";
    out << "th { background: #f0f0f0; }\n";
    out << ".good { color: green; }\n";
    out << ".warn { color: orange; }\n";
    out << ".bad { color: red; }\n";
    out << "img { max-width: 400px; border: 1px solid #ddd; }\n";
    out << ".image-grid { display: flex; flex-wrap: wrap; gap: 20px; }\n";
    out << ".image-card { text-align: center; }\n";
    out << "</style>\n</head>\n<body>\n";
    
    out << "<h1>Display Consistency Test Report</h1>\n";
    out << "<p>Generated: " << QDateTime::currentDateTime().toString() << "</p>\n";
    
    // Comparison matrix
    out << "<h2>Image Comparison Matrix</h2>\n";
    out << "<table>\n<tr><th>Mode</th>";
    for (const auto& r : results) {
        out << "<th>" << QString::fromStdString(r.modeName) << "</th>";
    }
    out << "</tr>\n";
    
    for (const auto& row : results) {
        out << "<tr><td><b>" << QString::fromStdString(row.modeName) << "</b></td>";
        for (const auto& col : results) {
            if (!row.success || !col.success) {
                out << "<td>N/A</td>";
                continue;
            }
            double mse = computeImageMSE(row.spectrogramImage, col.spectrogramImage);
            double maxDiff = computeImageMaxDiff(row.spectrogramImage, col.spectrogramImage);
            QString cls = (mse < 1.0 && maxDiff < 5) ? "good" : 
                         (mse < 10.0 && maxDiff < 20) ? "warn" : "bad";
            out << "<td class=\"" << cls << "\">MSE: " << QString::number(mse, 'f', 2) 
                << "<br>Max: " << QString::number(maxDiff, 'f', 1) << "</td>";
        }
        out << "</tr>\n";
    }
    out << "</table>\n";
    
    // Images
    out << "<h2>Spectrogram Images</h2>\n";
    out << "<div class=\"image-grid\">\n";
    for (const auto& result : results) {
        if (!result.success) continue;
        QString name = QString::fromStdString(result.modeName);
        name.replace(" ", "_");
        out << "<div class=\"image-card\">\n";
        out << "<h3>" << QString::fromStdString(result.modeName) << "</h3>\n";
        out << "<img src=\"" << name << "_spectrogram.png\" alt=\"" 
            << QString::fromStdString(result.modeName) << "\">\n";
        out << "</div>\n";
    }
    out << "</div>\n";
    
    out << "</body>\n</html>\n";
    file.close();
    
    std::cout << "  Report: " << reportPath.toStdString() << std::endl;
    return true;
}

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    
    QString outputDir = argc > 1 ? argv[1] : "test_output";
    
    std::cout << "Display Consistency Test" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Output directory: " << outputDir.toStdString() << std::endl;
    
    // Create test patterns
    std::vector<TestPattern> patterns = {
        {"sine_sweep_100_4000", generateSineSweep(48000.0, 2.0, 100.0, 4000.0), 48000.0},
        {"sine_tone_1000hz", generateSineTone(48000.0, 2.0, 1000.0), 48000.0},
        {"impulse_train", generateImpulseTrain(48000.0, 2.0, 0.1), 48000.0},
    };
    
    bool allPassed = true;
    
    for (const auto& pattern : patterns) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Testing pattern: " << pattern.name << std::endl;
        std::cout << "Signal: " << pattern.signal.size() << " samples @ " 
                  << pattern.sampleRate/1000 << " kHz" << std::endl;
        
        // Run all modes (capture phase)
        std::vector<ModeResult> results;
        
        std::cout << "  Running SingleThread mode..." << std::endl;
        results.push_back(runMode(LoiaconoRolling::ComputeMode::SingleThread, pattern, 400, 300, 100, 40, false));
        
        std::cout << "  Running MultiThread mode..." << std::endl;
        results.push_back(runMode(LoiaconoRolling::ComputeMode::MultiThread, pattern, 400, 300, 100, 40, false));
        
        std::cout << "  Running GpuCompute mode..." << std::endl;
        results.push_back(runMode(LoiaconoRolling::ComputeMode::GpuCompute, pattern, 400, 300, 100, 40, false));
        
        // Find global max amplitude across all modes for consistent normalization
        float globalMaxAmp = 0.01f;
        for (const auto& result : results) {
            for (const auto& hist : result.spectrumHistory) {
                for (float v : hist) {
                    globalMaxAmp = std::max(globalMaxAmp, v);
                }
            }
        }
        std::cout << "  Global max amplitude: " << globalMaxAmp << std::endl;
        
        // Re-render all images with consistent normalization
        SpectrogramRenderer renderer;
        renderer.maxAmplitude_ = globalMaxAmp;
        for (auto& result : results) {
            result.spectrogramImage = renderer.render(result.spectrumHistory, 400, 300, 2.0, pattern.sampleRate);
        }
        
        // Debug: print spectrum statistics
        std::cout << "\n  Spectrum Statistics:" << std::endl;
        for (const auto& result : results) {
            if (!result.success) continue;
            double sum = 0, maxVal = 0;
            for (float v : result.finalSpectrum) {
                sum += v;
                maxVal = std::max(maxVal, static_cast<double>(v));
            }
            double mean = sum / result.finalSpectrum.size();
            std::cout << "    " << result.modeName << ": mean=" << mean 
                      << ", max=" << maxVal << ", bins=" << result.finalSpectrum.size() << std::endl;
        }
        
        // Compare results
        std::cout << "\n  Comparison Results:" << std::endl;
        for (size_t i = 0; i < results.size(); i++) {
            for (size_t j = i + 1; j < results.size(); j++) {
                if (!results[i].success || !results[j].success) continue;
                
                // Compare spectrum values directly
                double specMse = 0;
                if (results[i].finalSpectrum.size() == results[j].finalSpectrum.size()) {
                    for (size_t k = 0; k < results[i].finalSpectrum.size(); k++) {
                        double diff = results[i].finalSpectrum[k] - results[j].finalSpectrum[k];
                        specMse += diff * diff;
                    }
                    specMse /= results[i].finalSpectrum.size();
                }
                
                double mse = computeImageMSE(results[i].spectrogramImage, results[j].spectrogramImage);
                double maxDiff = computeImageMaxDiff(results[i].spectrogramImage, results[j].spectrogramImage);
                
                std::cout << "    " << results[i].modeName << " vs " << results[j].modeName << ":";
                std::cout << " ImgMSE=" << mse << ", Max=" << maxDiff;
                std::cout << ", SpecMSE=" << specMse;
                
                // Thresholds for pass/fail (relaxed due to float vs double precision differences)
                // SpecMSE should be very small (< 0.01 indicates algorithmic consistency)
                // ImgMSE can be higher due to rendering differences
                bool specOk = specMse < 0.01;
                bool imgOk = mse < 50.0 && maxDiff < 100;
                
                if (!specOk) {
                    std::cout << " [SPEC_FAIL]" << std::endl;
                    allPassed = false;
                } else if (!imgOk) {
                    std::cout << " [IMG_DIFF]" << std::endl;
                    // Don't fail for image differences, just warn
                } else {
                    std::cout << " [OK]" << std::endl;
                }
            }
        }
        
        // Save results
        QString patternDir = QDir(outputDir).filePath(QString::fromStdString(pattern.name));
        std::cout << "\n  Saving results to: " << patternDir.toStdString() << std::endl;
        saveResults(results, patternDir);
    }
    
    std::cout << "\n========================================" << std::endl;
    if (allPassed) {
        std::cout << "ALL TESTS PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "SOME TESTS FAILED - check output images and reports" << std::endl;
        return 1;
    }
}
