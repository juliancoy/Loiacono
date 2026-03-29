// Spectrogram — scroll existing rows up by 1, write new row at bottom
// This is a compute shader that operates on a storage texture

struct SpectrogramParams {
    width: u32,       // number of frequency bins
    height: u32,      // spectrogram texture height in pixels
    max_amplitude: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: SpectrogramParams;
@group(0) @binding(1) var<storage, read> spectrum: array<f32>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<storage, read> prev_data: array<u32>;  // packed rgba from previous frame
@group(0) @binding(4) var<storage, read_write> curr_data: array<u32>;  // packed rgba for current frame

// Hot colormap: black → blue → cyan → green → yellow → red → white
fn hot_colormap(t_in: f32) -> vec4<f32> {
    let t = clamp(t_in, 0.0, 1.0);

    var r: f32; var g: f32; var b: f32;

    if (t < 0.1) {
        // black → dark blue
        let s = t / 0.1;
        r = 0.0; g = 0.0; b = s * 0.5;
    } else if (t < 0.3) {
        // dark blue → cyan
        let s = (t - 0.1) / 0.2;
        r = 0.0; g = s; b = 0.5 + s * 0.5;
    } else if (t < 0.5) {
        // cyan → green
        let s = (t - 0.3) / 0.2;
        r = 0.0; g = 1.0; b = 1.0 - s;
    } else if (t < 0.7) {
        // green → yellow
        let s = (t - 0.5) / 0.2;
        r = s; g = 1.0; b = 0.0;
    } else if (t < 0.9) {
        // yellow → red
        let s = (t - 0.7) / 0.2;
        r = 1.0; g = 1.0 - s; b = 0.0;
    } else {
        // red → white
        let s = (t - 0.9) / 0.1;
        r = 1.0; g = s; b = s;
    }

    return vec4<f32>(r, g, b, 1.0);
}

@compute @workgroup_size(128)
fn scroll_and_write(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let col = global_id.x;
    if (col >= params.width) {
        return;
    }

    // Scroll: copy rows from prev_data shifted up by 1
    for (var row: u32 = 0u; row < params.height - 1u; row++) {
        let src_idx = (row + 1u) * params.width + col;
        let dst_idx = row * params.width + col;
        let pixel = prev_data[src_idx];
        curr_data[dst_idx] = pixel;
        textureStore(output_texture, vec2<i32>(i32(col), i32(row)), unpack4x8unorm(pixel));
    }

    // Write new bottom row from spectrum data
    let amplitude = spectrum[col];
    let normalized = log(1.0 + amplitude) / log(1.0 + params.max_amplitude);
    let color = hot_colormap(normalized);
    let bottom_row = params.height - 1u;
    let packed = pack4x8unorm(color);
    curr_data[bottom_row * params.width + col] = packed;
    textureStore(output_texture, vec2<i32>(i32(col), i32(bottom_row)), color);
}

// Fullscreen quad vertex shader
struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Two-triangle fullscreen quad from vertex index
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );
    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var output: VertexOutput;
    output.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    output.uv = uvs[vertex_index];
    return output;
}

@group(0) @binding(0) var display_texture: texture_2d<f32>;
@group(0) @binding(1) var display_sampler: sampler;

@fragment
fn fragment_main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    return textureSample(display_texture, display_sampler, uv);
}
