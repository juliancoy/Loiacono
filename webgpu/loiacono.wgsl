// Loiacono Transform — WebGPU Compute Shader
// One workgroup per frequency bin, shared-memory reduction
// Direct port of the generic GLSL compute shader to WGSL

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

    // Each thread accumulates over strided samples within the analysis window
    var tr: f32 = 0.0;
    var ti: f32 = 0.0;

    var n = window_start + thread_ix;
    loop {
        if (n >= params.signal_length) {
            break;
        }
        let read_ix = (n + params.offset) % params.signal_length;
        let datum = x[read_ix];
        let angle = 2.0 * PI * this_f * f32(n);
        tr += datum * cos(angle) * dftlen;
        ti -= datum * sin(angle) * dftlen;
        n += WORKGROUP_SIZE;
    }

    // Workgroup-local parallel reduction via shared memory
    shared_tr[thread_ix] = tr;
    shared_ti[thread_ix] = ti;
    workgroupBarrier();

    // Tree reduction: 128 → 64 → 32 → 16 → 8 → 4 → 2 → 1
    var stride: u32 = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (thread_ix < stride) {
            shared_tr[thread_ix] += shared_tr[thread_ix + stride];
            shared_ti[thread_ix] += shared_ti[thread_ix + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    // Thread 0 writes the magnitude for this frequency bin
    if (thread_ix == 0u) {
        L[frequency_ix] = sqrt(shared_tr[0] * shared_tr[0] + shared_ti[0] * shared_ti[0]);
    }
}
