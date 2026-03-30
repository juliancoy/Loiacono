// Loiacono Transform — WebGPU Live Spectrogram (horizontal)
// Time scrolls left-to-right, frequency on Y axis (log scale)

// ─── Inline WGSL Shader ─────────────────────────────────────────
const LOIACONO_WGSL = /* wgsl */`
const PI: f32 = 3.14159265358979;

struct Params {
    signal_length: u32,
    num_frequencies: u32,
    multiple: u32,
    offset: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> f: array<f32>;
@group(0) @binding(3) var<storage, read_write> L: array<f32>;

const WORKGROUP_SIZE: u32 = 128;
var<workgroup> shared_tr: array<f32, 128>;
var<workgroup> shared_ti: array<f32, 128>;

@compute @workgroup_size(128)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let frequency_ix = workgroup_id.x;
    let thread_ix = local_id.x;
    let this_f = f[frequency_ix];
    let this_p = 1.0 / this_f;
    let window_len = u32(f32(params.multiple) * this_p);
    let window_start = params.signal_length - min(window_len, params.signal_length);
    let dftlen = 1.0 / sqrt(f32(params.multiple) / this_f);

    var tr: f32 = 0.0;
    var ti: f32 = 0.0;
    var n = window_start + thread_ix;
    loop {
        if (n >= params.signal_length) { break; }
        let read_ix = (n + params.offset) % params.signal_length;
        let datum = x[read_ix];
        let angle = 2.0 * PI * this_f * f32(n);
        tr += datum * cos(angle) * dftlen;
        ti -= datum * sin(angle) * dftlen;
        n += WORKGROUP_SIZE;
    }

    shared_tr[thread_ix] = tr;
    shared_ti[thread_ix] = ti;
    workgroupBarrier();

    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (thread_ix < stride) {
            shared_tr[thread_ix] += shared_tr[thread_ix + stride];
            shared_ti[thread_ix] += shared_ti[thread_ix + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (thread_ix == 0u) {
        L[frequency_ix] = sqrt(shared_tr[0] * shared_tr[0] + shared_ti[0] * shared_ti[0]);
    }
}
`;

// ─── Settings (live-adjustable) ──────────────────────────────────
const SAMPLE_RATE = 48000;
const SIGNAL_LENGTH = 1 << 15;
const SPECTROGRAM_WIDTH = 800;
const DISPLAY_HEIGHT = 400;

let multiple = 40;
let freqMin = 100;
let freqMax = 3000;
let freqBins = 200;
let fprime = [];
let numFreq = 0;

// Build log-spaced frequency bins
function rebuildFreqs() {
    fprime = [];
    const logMin = Math.log(freqMin);
    const logMax = Math.log(freqMax);
    const logStep = (logMax - logMin) / freqBins;
    for (let i = 0; i < freqBins; i++) {
        const f = Math.exp(logMin + i * logStep);
        fprime.push(f / SAMPLE_RATE);
    }
    numFreq = fprime.length;
    resizeCanvas();
    updateFreqLabels();
}

// ─── DOM ─────────────────────────────────────────────────────────
const canvas = document.getElementById("spectrogram");
const startBtn = document.getElementById("startBtn");
const testBtn = document.getElementById("testBtn");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");

const multipleSlider = document.getElementById("multipleSlider");
const multipleVal = document.getElementById("multipleVal");
const freqBinsSlider = document.getElementById("freqBinsSlider");
const freqBinsVal = document.getElementById("freqBinsVal");
const freqMinSlider = document.getElementById("freqMinSlider");
const freqMinVal = document.getElementById("freqMinVal");
const freqMaxSlider = document.getElementById("freqMaxSlider");
const freqMaxVal = document.getElementById("freqMaxVal");

let ctx2d;

function resizeCanvas() {
    canvas.width = SPECTROGRAM_WIDTH;
    canvas.height = numFreq;
    canvas.style.width = `${SPECTROGRAM_WIDTH}px`;
    canvas.style.height = `${DISPLAY_HEIGHT}px`;
    ctx2d = canvas.getContext("2d");
    ctx2d.fillStyle = "#000";
    ctx2d.fillRect(0, 0, canvas.width, canvas.height);
}

function updateFreqLabels() {
    const container = document.getElementById("freqLabelsY");
    container.innerHTML = "";
    // Log-spaced labels
    const labelFreqs = [100, 200, 500, 1000, 2000, 5000, 10000].filter(f => f >= freqMin && f <= freqMax);
    if (labelFreqs.length < 2) {
        // Fallback: linear labels
        for (let i = 0; i < 5; i++) {
            const f = freqMin + (freqMax - freqMin) * (i / 4);
            labelFreqs.push(Math.round(f));
        }
    }
    for (const f of labelFreqs) {
        const span = document.createElement("span");
        span.textContent = f >= 1000 ? `${(f/1000).toFixed(1)}k` : `${f}`;
        // Position based on log scale
        const logPos = (Math.log(f) - Math.log(freqMin)) / (Math.log(freqMax) - Math.log(freqMin));
        span.style.position = "absolute";
        span.style.bottom = `${logPos * 100}%`;
        span.style.transform = "translateY(50%)";
        container.appendChild(span);
    }
}

// ─── State ───────────────────────────────────────────────────────
let device;
let loiaconoPipeline, loiaconoBindGroup;
let signalBuffer, spectrumBuffer, freqBuffer, paramsBuffer, readbackBufferA;
let ringBuffer = new Float32Array(SIGNAL_LENGTH);
let ringOffset = 0;
let audioContext, audioSource, workletNode;
let isRunning = false;
let testOscillator = null;
let maxAmplitude = 1.0;
let frameCount = 0;
let lastFrameTime = 0;
let fps = 0;

// ─── Colormap ────────────────────────────────────────────────────
function hotColor(t) {
    t = Math.max(0, Math.min(1, t));
    t = Math.pow(t, 0.6);
    let r, g, b;
    if (t < 0.05) {
        r = 0; g = 0; b = 0;
    } else if (t < 0.2) {
        const s = (t - 0.05) / 0.15;
        r = 0; g = 0; b = Math.floor(s * 200);
    } else if (t < 0.4) {
        const s = (t - 0.2) / 0.2;
        r = 0; g = Math.floor(s * 255); b = Math.floor(200 + s * 55);
    } else if (t < 0.6) {
        const s = (t - 0.4) / 0.2;
        r = 0; g = 255; b = Math.floor(255 * (1 - s));
    } else if (t < 0.8) {
        const s = (t - 0.6) / 0.2;
        r = Math.floor(s * 255); g = 255; b = 0;
    } else {
        const s = (t - 0.8) / 0.2;
        r = 255; g = Math.floor(255 * (1 - s)); b = 0;
    }
    return [r, g, b, 255];
}

// ─── WebGPU Init ─────────────────────────────────────────────────
async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported. Try Chrome 113+.");
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("No WebGPU adapter found.");
    device = await adapter.requestDevice();

    statusEl.textContent = "Ready \u2014 click Start Microphone or Test Tone";
    startBtn.disabled = false;
    testBtn.disabled = false;
}

// ─── Spectrogram Drawing ─────────────────────────────────────────
function drawSpectrogramColumn(spectrum) {
    // Scroll existing image left by 1 pixel
    const imageData = ctx2d.getImageData(1, 0, canvas.width - 1, canvas.height);
    ctx2d.putImageData(imageData, 0, 0);

    // Auto-adjust max amplitude
    let currentMax = 0;
    for (let i = 0; i < spectrum.length; i++) {
        if (spectrum[i] > currentMax) currentMax = spectrum[i];
    }
    if (currentMax > maxAmplitude) {
        maxAmplitude = currentMax;
    } else {
        maxAmplitude = maxAmplitude * 0.998 + currentMax * 0.002;
    }
    maxAmplitude = Math.max(maxAmplitude, 0.1);

    // Draw new rightmost column (low freq at bottom)
    const col = ctx2d.createImageData(1, canvas.height);
    const logMax = Math.log(1 + maxAmplitude);
    for (let i = 0; i < numFreq && i < canvas.height; i++) {
        const canvasRow = canvas.height - 1 - i;
        const normalized = Math.log(1 + spectrum[i]) / logMax;
        const [r, g, b, a] = hotColor(normalized);
        col.data[canvasRow * 4] = r;
        col.data[canvasRow * 4 + 1] = g;
        col.data[canvasRow * 4 + 2] = b;
        col.data[canvasRow * 4 + 3] = a;
    }
    ctx2d.putImageData(col, canvas.width - 1, 0);
}

// ─── Audio Capture ───────────────────────────────────────────────
async function startAudio() {
    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });

    const workletCode = `
        class RingBufferProcessor extends AudioWorkletProcessor {
            process(inputs) {
                const input = inputs[0];
                if (input.length > 0) {
                    this.port.postMessage(input[0]);
                }
                return true;
            }
        }
        registerProcessor("ring-buffer-processor", RingBufferProcessor);
    `;
    const blob = new Blob([workletCode], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    await audioContext.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioSource = audioContext.createMediaStreamSource(stream);
    workletNode = new AudioWorkletNode(audioContext, "ring-buffer-processor");

    workletNode.port.onmessage = (e) => {
        const samples = e.data;
        for (let i = 0; i < samples.length; i++) {
            ringBuffer[ringOffset] = samples[i];
            ringOffset = (ringOffset + 1) % SIGNAL_LENGTH;
        }
    };

    audioSource.connect(workletNode);
    workletNode.connect(audioContext.destination);
}

function startTestTone() {
    if (!audioContext) {
        audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    }

    const genSamples = () => {
        const chunkSize = 256;
        for (let i = 0; i < chunkSize; i++) {
            const t = ringOffset / SAMPLE_RATE;
            ringBuffer[ringOffset] =
                0.3 * Math.sin(2 * Math.PI * 440 * t) +
                0.2 * Math.sin(2 * Math.PI * 880 * t) +
                0.15 * Math.sin(2 * Math.PI * 1320 * t) +
                0.1 * Math.sin(2 * Math.PI * 1760 * t);
            ringOffset = (ringOffset + 1) % SIGNAL_LENGTH;
        }
    };
    for (let i = 0; i < SIGNAL_LENGTH / 256; i++) genSamples();
    testOscillator = setInterval(genSamples, 5);
}

// ─── CPU Loiacono ────────────────────────────────────────────────
function cpuLoiacono(signal, freqs, mult, signalLen) {
    const spectrum = new Float32Array(freqs.length);
    const MAX_WINDOW = 4096;
    for (let fi = 0; fi < freqs.length; fi++) {
        const thisF = freqs[fi];
        const thisP = 1.0 / thisF;
        const windowLen = Math.min(Math.floor(mult * thisP), signalLen, MAX_WINDOW);
        const windowStart = signalLen - windowLen;
        const dftlen = 1.0 / Math.sqrt(windowLen);
        let tr = 0, ti = 0;
        for (let n = windowStart; n < signalLen; n++) {
            const datum = signal[n];
            const angle = 2 * Math.PI * thisF * n;
            tr += datum * Math.cos(angle) * dftlen;
            ti -= datum * Math.sin(angle) * dftlen;
        }
        spectrum[fi] = Math.sqrt(tr * tr + ti * ti);
    }
    return spectrum;
}

// ─── Frame Loop (full speed) ─────────────────────────────────────
function runLoop() {
    if (!isRunning) return;

    try {
        const t0 = performance.now();
        const spectrum = cpuLoiacono(ringBuffer, fprime, multiple, SIGNAL_LENGTH);
        const computeMs = performance.now() - t0;

        drawSpectrogramColumn(spectrum);
        frameCount++;

        // FPS tracking
        const now = performance.now();
        if (now - lastFrameTime > 500) {
            fps = Math.round(frameCount / ((now - lastFrameTime) / 1000));
            frameCount = 0;
            lastFrameTime = now;

            let peak = 0, peakIdx = 0;
            for (let i = 0; i < spectrum.length; i++) {
                if (spectrum[i] > peak) { peak = spectrum[i]; peakIdx = i; }
            }
            const peakFreq = (fprime[peakIdx] * SAMPLE_RATE).toFixed(0);
            statusEl.textContent = `Peak: ${peakFreq} Hz | ${fps} fps | ${computeMs.toFixed(1)}ms/frame | ${numFreq} bins`;
        }
    } catch (e) {
        statusEl.textContent = "Error: " + e.message;
        console.error(e);
        return;
    }

    requestAnimationFrame(runLoop);
}

// ─── Settings Handlers ───────────────────────────────────────────
multipleSlider.addEventListener("input", () => {
    multiple = parseInt(multipleSlider.value);
    multipleVal.textContent = multiple;
});

freqBinsSlider.addEventListener("input", () => {
    freqBins = parseInt(freqBinsSlider.value);
    freqBinsVal.textContent = freqBins;
    rebuildFreqs();
});

freqMinSlider.addEventListener("input", () => {
    freqMin = parseInt(freqMinSlider.value);
    freqMinVal.textContent = freqMin;
    if (freqMin >= freqMax - 100) {
        freqMax = freqMin + 100;
        freqMaxSlider.value = freqMax;
        freqMaxVal.textContent = freqMax;
    }
    rebuildFreqs();
});

freqMaxSlider.addEventListener("input", () => {
    freqMax = parseInt(freqMaxSlider.value);
    freqMaxVal.textContent = freqMax;
    if (freqMax <= freqMin + 100) {
        freqMin = freqMax - 100;
        freqMinSlider.value = freqMin;
        freqMinVal.textContent = freqMin;
    }
    rebuildFreqs();
});

// ─── UI Handlers ─────────────────────────────────────────────────
startBtn.addEventListener("click", async () => {
    if (isRunning && !testOscillator) {
        isRunning = false;
        if (audioContext) { audioContext.close(); audioContext = null; }
        startBtn.textContent = "Start Microphone";
        startBtn.classList.remove("active");
        statusEl.textContent = "Stopped";
        return;
    }

    try {
        if (testOscillator) {
            clearInterval(testOscillator);
            testOscillator = null;
            testBtn.classList.remove("active");
            testBtn.textContent = "Test Tone (A440)";
        }
        await startAudio();
        isRunning = true;
        lastFrameTime = performance.now();
        frameCount = 0;
        startBtn.textContent = "Stop";
        startBtn.classList.add("active");
        statusEl.textContent = "Listening...";
        runLoop();
    } catch (e) {
        showError("Microphone error: " + e.message);
    }
});

testBtn.addEventListener("click", () => {
    if (testOscillator) {
        clearInterval(testOscillator);
        testOscillator = null;
        isRunning = false;
        testBtn.textContent = "Test Tone (A440)";
        testBtn.classList.remove("active");
        statusEl.textContent = "Ready";
        return;
    }
    startTestTone();
    isRunning = true;
    lastFrameTime = performance.now();
    frameCount = 0;
    testBtn.textContent = "Stop Tone";
    testBtn.classList.add("active");
    statusEl.textContent = "Playing A440 + harmonics...";
    runLoop();
});

function showError(msg) {
    errorEl.textContent = msg;
    errorEl.style.display = "block";
    statusEl.textContent = "Error";
}

// ─── Boot ────────────────────────────────────────────────────────
rebuildFreqs();
initWebGPU().catch((e) => {
    showError(e.message);
    console.error(e);
});
