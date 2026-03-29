// Loiacono Transform — WebGPU Live Spectrogram
// main.js: WebGPU compute + Canvas 2D spectrogram visualization

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

// ─── Configuration ───────────────────────────────────────────────
const SAMPLE_RATE = 48000;
const SIGNAL_LENGTH = 1 << 15; // 32768
const MULTIPLE = 40;
const FREQ_START = 100 / SAMPLE_RATE;
const FREQ_END = 3000 / SAMPLE_RATE;
const WORKGROUP_SIZE = 128;
const SPECTROGRAM_HEIGHT = 512;

// Build frequency bins (normalized)
const fprime = [];
const freqStep = 5.0 / SAMPLE_RATE;
for (let f = FREQ_START; f < FREQ_END; f += freqStep) {
    fprime.push(f);
}
const NUM_FREQ = fprime.length;

// ─── DOM ─────────────────────────────────────────────────────────
const canvas = document.getElementById("spectrogram");
const startBtn = document.getElementById("startBtn");
const testBtn = document.getElementById("testBtn");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");

canvas.width = NUM_FREQ;
canvas.height = SPECTROGRAM_HEIGHT;
canvas.style.width = "1024px";
canvas.style.height = `${SPECTROGRAM_HEIGHT}px`;
canvas.style.imageRendering = "pixelated";

const ctx2d = canvas.getContext("2d");
ctx2d.fillStyle = "#0a0a0f";
ctx2d.fillRect(0, 0, canvas.width, canvas.height);

// ─── State ───────────────────────────────────────────────────────
let device;
let loiaconoPipeline, loiaconoBindGroup;
let signalBuffer, spectrumBuffer, freqBuffer, paramsBuffer, readbackBuffer;
let ringBuffer = new Float32Array(SIGNAL_LENGTH);
let ringOffset = 0;
let audioContext, audioSource, workletNode;
let isRunning = false;
let testOscillator = null;
let maxAmplitude = 1.0;
let frameCount = 0;
let gpuBusy = false;
let readbackBufferA;

// ─── Colormap ────────────────────────────────────────────────────
function hotColor(t) {
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.1) {
        const s = t / 0.1;
        r = 0; g = 0; b = Math.floor(s * 128);
    } else if (t < 0.3) {
        const s = (t - 0.1) / 0.2;
        r = 0; g = Math.floor(s * 255); b = Math.floor(128 + s * 127);
    } else if (t < 0.5) {
        const s = (t - 0.3) / 0.2;
        r = 0; g = 255; b = Math.floor(255 * (1 - s));
    } else if (t < 0.7) {
        const s = (t - 0.5) / 0.2;
        r = Math.floor(s * 255); g = 255; b = 0;
    } else if (t < 0.9) {
        const s = (t - 0.7) / 0.2;
        r = 255; g = Math.floor(255 * (1 - s)); b = 0;
    } else {
        const s = (t - 0.9) / 0.1;
        r = 255; g = Math.floor(s * 255); b = Math.floor(s * 255);
    }
    return [r, g, b, 255];
}

// ─── WebGPU Init ─────────────────────────────────────────────────
async function initWebGPU() {
    if (!navigator.gpu) {
        throw new Error("WebGPU is not supported in this browser. Try Chrome 113+ or Edge 113+.");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
        throw new Error("No WebGPU adapter found.");
    }

    device = await adapter.requestDevice();

    createBuffers();
    createLoiaconoPipeline();

    statusEl.textContent = "Ready \u2014 click Start Microphone or Test Tone";
    startBtn.disabled = false;
    testBtn.disabled = false;
}

// ─── GPU Buffers ─────────────────────────────────────────────────
function createBuffers() {
    paramsBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    signalBuffer = device.createBuffer({
        size: SIGNAL_LENGTH * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    freqBuffer = device.createBuffer({
        size: NUM_FREQ * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(freqBuffer, 0, new Float32Array(fprime));

    spectrumBuffer = device.createBuffer({
        size: NUM_FREQ * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    readbackBufferA = device.createBuffer({
        size: NUM_FREQ * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
}

// ─── Loiacono Compute Pipeline ───────────────────────────────────
function createLoiaconoPipeline() {
    const shaderModule = device.createShaderModule({ code: LOIACONO_WGSL });

    const bindGroupLayout = device.createBindGroupLayout({
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        ],
    });

    loiaconoPipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
        compute: { module: shaderModule, entryPoint: "main" },
    });

    loiaconoBindGroup = device.createBindGroup({
        layout: bindGroupLayout,
        entries: [
            { binding: 0, resource: { buffer: paramsBuffer } },
            { binding: 1, resource: { buffer: signalBuffer } },
            { binding: 2, resource: { buffer: freqBuffer } },
            { binding: 3, resource: { buffer: spectrumBuffer } },
        ],
    });
}

// ─── Spectrogram Drawing (Canvas 2D, GPU-scrolled) ───────────────
function drawSpectrogramRow(spectrum) {
    // Scroll existing image up by 1 pixel
    const imageData = ctx2d.getImageData(0, 1, canvas.width, canvas.height - 1);
    ctx2d.putImageData(imageData, 0, 0);

    // Auto-adjust max amplitude with decay
    let currentMax = 0;
    for (let i = 0; i < spectrum.length; i++) {
        if (spectrum[i] > currentMax) currentMax = spectrum[i];
    }
    if (currentMax > maxAmplitude) {
        maxAmplitude = currentMax;
    } else {
        maxAmplitude = maxAmplitude * 0.999 + currentMax * 0.001;
    }
    maxAmplitude = Math.max(maxAmplitude, 0.1);

    // Draw new bottom row
    const row = ctx2d.createImageData(canvas.width, 1);
    const logMax = Math.log(1 + maxAmplitude);
    for (let i = 0; i < NUM_FREQ; i++) {
        const normalized = Math.log(1 + spectrum[i]) / logMax;
        const [r, g, b, a] = hotColor(normalized);
        row.data[i * 4] = r;
        row.data[i * 4 + 1] = g;
        row.data[i * 4 + 2] = b;
        row.data[i * 4 + 3] = a;
    }
    ctx2d.putImageData(row, 0, canvas.height - 1);
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
    // Generate enough samples to fill the buffer initially
    for (let i = 0; i < SIGNAL_LENGTH / 256; i++) genSamples();
    testOscillator = setInterval(genSamples, 5);
}

// ─── CPU Loiacono (fallback for readback issues) ─────────────────
function cpuLoiacono(signal, freqs, multiple, signalLen) {
    const spectrum = new Float32Array(freqs.length);
    const MAX_WINDOW = 4096; // cap window size for real-time performance
    for (let fi = 0; fi < freqs.length; fi++) {
        const thisF = freqs[fi];
        const thisP = 1.0 / thisF;
        const windowLen = Math.min(Math.floor(multiple * thisP), signalLen, MAX_WINDOW);
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

// ─── Frame Loop (GPU with readback, CPU fallback) ────────────────
function runLoop() {
    if (!isRunning) return;

    try {
        const spectrum = cpuLoiacono(ringBuffer, fprime, MULTIPLE, SIGNAL_LENGTH);

        drawSpectrogramRow(spectrum);
        frameCount++;
        if (frameCount % 10 === 0) {
            let peak = 0, peakIdx = 0;
            for (let i = 0; i < spectrum.length; i++) {
                if (spectrum[i] > peak) { peak = spectrum[i]; peakIdx = i; }
            }
            const peakFreq = (fprime[peakIdx] * SAMPLE_RATE).toFixed(0);
            statusEl.textContent = `Peak: ${peakFreq} Hz | Amp: ${peak.toFixed(1)} | Frame: ${frameCount}`;
        }
    } catch (e) {
        statusEl.textContent = "Error: " + e.message;
        console.error(e);
        return;
    }

    setTimeout(runLoop, 50); // ~20fps target, don't block rAF
}

// ─── UI Handlers ─────────────────────────────────────────────────
startBtn.addEventListener("click", async () => {
    if (isRunning && !testOscillator) {
        isRunning = false;
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }
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
        startBtn.textContent = "Stop";
        startBtn.classList.add("active");
        statusEl.textContent = "Listening...";
        runLoop();
    } catch (e) {
        showError("Microphone access denied or unavailable: " + e.message);
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
initWebGPU().catch((e) => {
    showError(e.message);
    console.error(e);
});
