// Must match DofParams in src/mod.cpp (80 bytes).
struct P {
    maxRadiusFrac: f32,   // max blur radius as a fraction of target height
    strength: f32,
    focusDist: f32,
    focusRange: f32,
    farFalloff: f32,
    nearFalloff: f32,
    nearZ: f32,
    farZ: f32,
    reversedZ: f32,
    nearBlur: f32,
    tapCount: f32,        // maximum number of taps; the actual count scales with the blur radius
    ambientFarDistance: f32,
    ambientStrength: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
    rect: vec4<f32>,      // 3D viewport in 0..1 UV: x0, y0, x1, y1
};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var colorTex: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_2d<f32>;   // R32Float, textureLoad only
@group(0) @binding(3) var samp: sampler;

struct VO {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

var<private> tri: array<vec2<f32>, 3> = array<vec2<f32>, 3>(
    vec2(-1.0, 1.0), vec2(-1.0, -3.0), vec2(3.0, 1.0));

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VO {
    var o: VO;
    let v = tri[i];
    o.pos = vec4<f32>(v, 0.0, 1.0);
    o.uv = vec2<f32>(v.x * 0.5 + 0.5, 0.5 - v.y * 0.5);
    return o;
}

fn linZ(d0: f32) -> f32 {
    let d = select(d0, 1.0 - d0, p.reversedZ > 0.5);
    return p.nearZ * p.farZ / max(p.farZ - d * (p.farZ - p.nearZ), 1e-6);
}

// signed CoC in pixels: + far, - near
fn cocFromZ(z: f32, maxR: f32) -> f32 {
    let farT  = clamp((z - (p.focusDist + p.focusRange)) / max(p.farFalloff, 1e-3), 0.0, 1.0);
    let nearT = clamp(((p.focusDist - p.focusRange) - z) / max(p.nearFalloff, 1e-3), 0.0, 1.0);
    let focusCoc = (farT - select(0.0, nearT, p.nearBlur > 0.5)) * maxR * p.strength;

    // Ambient blur uses absolute camera distance, so it does not follow the attention point.
    // A half-threshold ramp keeps the transition gradual while limiting the effect to distant objects.
    let ambientT = clamp((z - p.ambientFarDistance) / max(p.ambientFarDistance * 0.5, 1.0), 0.0, 1.0);
    let ambientCoc = ambientT * maxR * p.ambientStrength;

    // Whichever contributes the larger blur wins, so a strong focus blur is never reduced
    // by the (weaker) ambient ramp in the far field.
    return select(focusCoc, ambientCoc, abs(ambientCoc) > abs(focusCoc));
}

fn loadCoc(px: vec2<i32>, maxR: f32) -> f32 {
    let dims = vec2<i32>(textureDimensions(depthTex));
    let d = textureLoad(depthTex, clamp(px, vec2<i32>(0), dims - vec2<i32>(1)), 0).r;
    return cocFromZ(linZ(d), maxR);
}

// Outputs PREMULTIPLIED color (rgb * a, a). With premultiplied alpha, bilinear filtering of the
// half-resolution target is correct at mask edges (no dark fringes), and the blend state is
// One / OneMinusSrcAlpha everywhere the result is blended over the scene.
@fragment
fn fs_main(v: VO) -> @location(0) vec4<f32> {
    if (v.uv.x < p.rect.x || v.uv.y < p.rect.y || v.uv.x > p.rect.z || v.uv.y > p.rect.w) {
        discard;
    }
    let colorDims = vec2<f32>(textureDimensions(colorTex));
    let texel = 1.0 / colorDims;
    let maxR = p.maxRadiusFrac * colorDims.y;

    let dims = vec2<f32>(textureDimensions(depthTex));
    let px = vec2<i32>(v.uv * dims);
    let cc = loadCoc(px, maxR);
    let r = abs(cc);
    if (r < 0.5) { discard; }

    const TAU = 6.2831853;
    const CG = -0.7373689;   // cos(golden angle)
    const SG =  0.6754903;   // sin(golden angle)
    // These values come from Jorge Jimenez's "NEXT GENERATION POST PROCESSING IN CALL OF DUTY: ADVANCED WARFARE" SIGGRAPH 2014 course notes.
    // (more info: https://blog.demofox.org/2022/01/01/interleaved-gradient-noise-a-different-kind-of-low-discrepancy-sequence/)
    let rot = TAU * fract(52.9829189 * fract(dot(v.pos.xy, vec2<f32>(0.06711056, 0.00583715))));
    var acc = textureSampleLevel(colorTex, samp, v.uv, 0.0).rgb;
    var wsum = 1.0;
    var c = cos(rot);
    var s = sin(rot);

    // Adaptive tap count: small blurs need few samples, large blurs get the full budget.
    let maxN = i32(clamp(p.tapCount, 4.0, 64.0));
    let N = clamp(i32(ceil(r * 1.5)), 4, maxN);
    for (var i = 0; i < N; i++) {
        let rr = sqrt((f32(i) + 0.5) / f32(N));
        let o = vec2<f32>(c, s) * rr * r;
        let tc = loadCoc(px + vec2<i32>(round(o)), maxR);
        var w = clamp(abs(tc) - length(o) + 1.0, 0.0, 1.0);
        // sharp/foreground taps must not leak onto a farther, blurrier center
        if (cc > 0.0 && tc < cc) { w *= clamp(tc / cc, 0.0, 1.0); }
        acc += textureSampleLevel(colorTex, samp, v.uv + o * texel, 0.0).rgb * w;
        wsum += w;
        let nc = c * CG - s * SG;
        s = c * SG + s * CG;
        c = nc;
    }
    let a = smoothstep(0.5, 2.0, r);
    return vec4<f32>((acc / wsum) * a, a);
}

// Half-resolution path: the gathered target already holds premultiplied color.
@fragment
fn fs_composite(v: VO) -> @location(0) vec4<f32> {
    return textureSampleLevel(colorTex, samp, v.uv, 0.0);
}
