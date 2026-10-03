// High quality depth of field mod.
//
// Lets the game's drawDepth2 run (with the built-in depth of field set to Off, it updates the
// smoothed focus state in g_env_light and returns), then in a post-hook recomputes the focus
// parameters, snapshots color + depth, and records a fullscreen gather pass (res/dof_hq.wgsl).

#include "global.h"

#include "d/actor/d_a_player.h"
#include "d/d_camera.h"
#include "d/d_com_inf_game.h"
#include "d/d_kankyo.h"
#include "f_op/f_op_camera_mng.h"
#include "f_op/f_op_view.h"
#include "m_Do/m_Do_graphic.h"

#include "mods/service.hpp"
#include "mods/svc/config.h"
#include "mods/svc/gfx.h"
#include "mods/svc/hook.h"
#include "mods/svc/hook.hpp"
#include "mods/svc/log.h"
#include "mods/svc/resource.h"
#include "mods/svc/ui.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <webgpu/webgpu.h>

DEFINE_MOD();
IMPORT_SERVICE(ConfigService, svc_config);
IMPORT_SERVICE(GfxService, svc_gfx);
IMPORT_SERVICE(HookService, svc_hook);
IMPORT_SERVICE(LogService, svc_log);
IMPORT_SERVICE(ResourceService, svc_resource);
IMPORT_SERVICE(UiService, svc_ui);

namespace
{

    // ---------------------------------------------------------------------------------------------
    // Config
    // ---------------------------------------------------------------------------------------------
    ConfigVarHandle g_cvarEnabled = 0;            // enable/disable the effect
    ConfigVarHandle g_cvarIntensity = 0;          // % multiplier on the computed strength
    ConfigVarHandle g_cvarAmbientEnabled = 0;     // enable ambient blur in the far field
    ConfigVarHandle g_cvarAmbientFarBlur = 0;     // % minimum strength in the far field, even when focus is weak
    ConfigVarHandle g_cvarAmbientFarDistance = 0; // world-space distance where ambient blur begins
    ConfigVarHandle g_cvarMaxBlur = 0;            // tenths of a percent of target height (10 = 1.0%)
    ConfigVarHandle g_cvarFocusRange = 0;         // % of focus distance
    ConfigVarHandle g_cvarMinFocusRange = 0;      // world units
    ConfigVarHandle g_cvarFarFalloff = 0;         // % of focus distance
    ConfigVarHandle g_cvarNearFalloff = 0;        // % of focus distance
    ConfigVarHandle g_cvarTapCount = 0;           // number of taps in the gather pass
    ConfigVarHandle g_cvarHalfResolution = 0;     // run the gather pass at half resolution (faster, but blurrier)
    UiWindowHandle g_controlsWindow = 0;

    int64_t get_int_option(ConfigVarHandle handle, int64_t fallback)
    {
        int64_t value = fallback;
        if (handle == 0 || svc_config->get_int(mod_ctx, handle, &value) != MOD_OK)
            return fallback;

        return value;
    }

    bool get_bool_option(ConfigVarHandle handle, bool fallback)
    {
        bool value = fallback;
        if (handle == 0 || svc_config->get_bool(mod_ctx, handle, &value) != MOD_OK)
            return fallback;

        return value;
    }

    ModResult register_bool_option(const char *name, bool defaultValue, ConfigVarHandle &outHandle, ModError *error)
    {
        ConfigVarDesc desc = CONFIG_VAR_DESC_INIT;
        desc.name = name;
        desc.type = CONFIG_VAR_BOOL;
        desc.default_bool = defaultValue;
        if (svc_config->register_var(mod_ctx, &desc, &outHandle) != MOD_OK) {
            return mods::set_error(error, MOD_ERROR, "failed to register depth of field option");
        }
        return MOD_OK;
    }

    ModResult register_int_option(
        const char *name, int64_t defaultValue, ConfigVarHandle &outHandle, ModError *error)
    {
        ConfigVarDesc desc = CONFIG_VAR_DESC_INIT;
        desc.name = name;
        desc.type = CONFIG_VAR_INT;
        desc.default_int = defaultValue;
        if (svc_config->register_var(mod_ctx, &desc, &outHandle) != MOD_OK) {
            return mods::set_error(error, MOD_ERROR, "failed to register depth of field option");
        }
        return MOD_OK;
    }

    // ---------------------------------------------------------------------------------------------
    // GPU state (render worker thread, except init/shutdown)
    // ---------------------------------------------------------------------------------------------
    GfxDeviceInfo g_deviceInfo = GFX_DEVICE_INFO_INIT;
    GfxDrawTypeHandle g_drawType = 0;
    ResourceBuffer g_shaderSource = RESOURCE_BUFFER_INIT;

    WGPUShaderModule g_module = nullptr;
    WGPUBindGroupLayout g_bgl = nullptr;
    WGPUPipelineLayout g_pipelineLayout = nullptr;
    WGPUBindGroupLayout g_compositeBgl = nullptr;
    WGPUPipelineLayout g_compositePipelineLayout = nullptr;
    WGPUSampler g_sampler = nullptr;
    WGPURenderPipeline g_gatherPipeline = nullptr;
    WGPURenderPipeline g_compositePipeline = nullptr;
    GfxRenderTargetLayout g_gatherTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;
    GfxRenderTargetLayout g_sceneTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;

    // Mirror of the WGSL struct P (keep in sync with res/dof_hq.wgsl).
    struct DofParams
    {
        float maxRadiusFrac;
        float strength;
        float focusDist;
        float focusRange;
        float farFalloff;
        float nearFalloff;
        float nearZ;
        float farZ;
        float reversedZ;
        float nearBlur;
        float tapCount;
        float ambientFarDistance;
        float ambientStrength;
        float _pad[3];
        float rect[4];
    };
    static_assert(sizeof(DofParams) == 80);

    struct DrawPayload
    {
        WGPUTextureView color; // frame-pooled
        WGPUTextureView depth; // frame-pooled for gather, unused for composite
        uint32_t uniform_offset;
        uint32_t uniform_size;
        uint32_t composite;
    };
    static_assert(sizeof(DrawPayload) <= GFX_INLINE_DRAW_PAYLOAD_SIZE);
    static_assert(std::is_trivially_copyable_v<DrawPayload>);

    void release_pipeline()
    {
        if (g_gatherPipeline != nullptr) {
            wgpuRenderPipelineRelease(g_gatherPipeline);
            g_gatherPipeline = nullptr;
        }
        if (g_compositePipeline != nullptr) {
            wgpuRenderPipelineRelease(g_compositePipeline);
            g_compositePipeline = nullptr;
        }
        g_gatherTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;
        g_sceneTargetLayout = GFX_RENDER_TARGET_LAYOUT_INIT;
    }

    void release_gpu()
    {
        release_pipeline();
        if (g_sampler != nullptr) {
            wgpuSamplerRelease(g_sampler);
            g_sampler = nullptr;
        }
        if (g_pipelineLayout != nullptr) {
            wgpuPipelineLayoutRelease(g_pipelineLayout);
            g_pipelineLayout = nullptr;
        }
        if (g_compositePipelineLayout != nullptr) {
            wgpuPipelineLayoutRelease(g_compositePipelineLayout);
            g_compositePipelineLayout = nullptr;
        }
        if (g_bgl != nullptr) {
            wgpuBindGroupLayoutRelease(g_bgl);
            g_bgl = nullptr;
        }
        if (g_compositeBgl != nullptr) {
            wgpuBindGroupLayoutRelease(g_compositeBgl);
            g_compositeBgl = nullptr;
        }
        if (g_module != nullptr) {
            wgpuShaderModuleRelease(g_module);
            g_module = nullptr;
        }
    }

    bool init_gpu()
    {
        WGPUDevice device = g_deviceInfo.device;

        WGPUShaderSourceWGSL wgsl = WGPU_SHADER_SOURCE_WGSL_INIT;
        wgsl.code = {static_cast<const char *>(g_shaderSource.data), g_shaderSource.size};
        WGPUShaderModuleDescriptor moduleDesc = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        moduleDesc.nextInChain = &wgsl.chain;
        moduleDesc.label = {"dof_hq", WGPU_STRLEN};
        g_module = wgpuDeviceCreateShaderModule(device, &moduleDesc);
        if (g_module == nullptr)
            return false;

        // Explicit layout: the depth snapshot is R32Float, which must be bound as unfilterable.
        WGPUBindGroupLayoutEntry entries[4] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                               WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                               WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                               WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Fragment;
        entries[0].buffer.type = WGPUBufferBindingType_Uniform;
        entries[0].buffer.minBindingSize = sizeof(DofParams);

        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Fragment;
        entries[1].texture.sampleType = WGPUTextureSampleType_Float;
        entries[1].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Fragment;
        entries[2].texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
        entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Fragment;
        entries[3].sampler.type = WGPUSamplerBindingType_Filtering;

        WGPUBindGroupLayoutDescriptor bglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        bglDesc.label = {"dof_hq bgl", WGPU_STRLEN};
        bglDesc.entryCount = 4;
        bglDesc.entries = entries;
        g_bgl = wgpuDeviceCreateBindGroupLayout(device, &bglDesc);
        if (g_bgl == nullptr)
            return false;

        WGPUBindGroupLayoutEntry compositeEntries[2] = {WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT,
                                                        WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT};
        compositeEntries[0].binding = 1;
        compositeEntries[0].visibility = WGPUShaderStage_Fragment;
        compositeEntries[0].texture.sampleType = WGPUTextureSampleType_Float;
        compositeEntries[0].texture.viewDimension = WGPUTextureViewDimension_2D;

        compositeEntries[1].binding = 3;
        compositeEntries[1].visibility = WGPUShaderStage_Fragment;
        compositeEntries[1].sampler.type = WGPUSamplerBindingType_Filtering;

        WGPUBindGroupLayoutDescriptor compositeBglDesc = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        compositeBglDesc.label = {"dof_hq composite bgl", WGPU_STRLEN};
        compositeBglDesc.entryCount = 2;
        compositeBglDesc.entries = compositeEntries;
        g_compositeBgl = wgpuDeviceCreateBindGroupLayout(device, &compositeBglDesc);
        if (g_compositeBgl == nullptr)
            return false;

        WGPUPipelineLayoutDescriptor plDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        plDesc.label = {"dof_hq pipeline layout", WGPU_STRLEN};
        plDesc.bindGroupLayoutCount = 1;
        plDesc.bindGroupLayouts = &g_bgl;
        g_pipelineLayout = wgpuDeviceCreatePipelineLayout(device, &plDesc);
        if (g_pipelineLayout == nullptr)
            return false;

        WGPUPipelineLayoutDescriptor compositePlDesc = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        compositePlDesc.label = {"dof_hq composite pipeline layout", WGPU_STRLEN};
        compositePlDesc.bindGroupLayoutCount = 1;
        compositePlDesc.bindGroupLayouts = &g_compositeBgl;
        g_compositePipelineLayout = wgpuDeviceCreatePipelineLayout(device, &compositePlDesc);
        if (g_compositePipelineLayout == nullptr)
            return false;

        WGPUSamplerDescriptor samplerDesc = WGPU_SAMPLER_DESCRIPTOR_INIT;
        samplerDesc.label = {"dof_hq sampler", WGPU_STRLEN};
        samplerDesc.magFilter = WGPUFilterMode_Linear;
        samplerDesc.minFilter = WGPUFilterMode_Linear;
        g_sampler = wgpuDeviceCreateSampler(device, &samplerDesc);
        return g_sampler != nullptr;
    }

    bool build_pipeline(const GfxRenderTargetLayout &layout, bool composite, WGPURenderPipeline &pipeline)
    {
        WGPUBlendState blendState{
            .color = {.operation = WGPUBlendOperation_Add,
                      .srcFactor = WGPUBlendFactor_SrcAlpha,
                      .dstFactor = WGPUBlendFactor_OneMinusSrcAlpha},
            .alpha = {.operation = WGPUBlendOperation_Add,
                      .srcFactor = WGPUBlendFactor_Zero,
                      .dstFactor = WGPUBlendFactor_One}, // keep destination alpha
        };
        WGPUColorTargetState colorTargets[GFX_MAX_COLOR_ATTACHMENTS];
        const uint32_t colorTargetCount = gfx_init_color_target_states(
            &layout, colorTargets, composite ? &blendState : nullptr,
            static_cast<WGPUColorWriteMask>(
                composite ? (WGPUColorWriteMask_Red | WGPUColorWriteMask_Green |
                             WGPUColorWriteMask_Blue)
                          : WGPUColorWriteMask_All));

        WGPUFragmentState fragment = WGPU_FRAGMENT_STATE_INIT;
        fragment.module = g_module;
        fragment.entryPoint = {composite ? "fs_composite" : "fs_main", WGPU_STRLEN};
        fragment.targetCount = colorTargetCount;
        fragment.targets = colorTargets;

        WGPUDepthStencilState depthStencil = WGPU_DEPTH_STENCIL_STATE_INIT;
        depthStencil.format = layout.depth_stencil_format;
        depthStencil.depthWriteEnabled = WGPUOptionalBool_False;
        depthStencil.depthCompare = WGPUCompareFunction_Always;

        WGPURenderPipelineDescriptor pipelineDesc = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
        pipelineDesc.label = {composite ? "dof_hq composite" : "dof_hq gather", WGPU_STRLEN};
        pipelineDesc.layout = composite ? g_compositePipelineLayout : g_pipelineLayout;
        pipelineDesc.vertex.module = g_module;
        pipelineDesc.vertex.entryPoint = {"vs_main", WGPU_STRLEN};
        pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
        pipelineDesc.depthStencil = &depthStencil;
        pipelineDesc.multisample.count = layout.sample_count;
        pipelineDesc.fragment = &fragment;
        pipeline = wgpuDeviceCreateRenderPipeline(g_deviceInfo.device, &pipelineDesc);
        return pipeline != nullptr;
    }

    // Scene and offscreen attachments can change at runtime, so rebuild lazily on layout.key changes.
    bool ensure_pipeline(const GfxRenderTargetLayout &layout, bool composite)
    {
        WGPURenderPipeline &pipeline = composite ? g_compositePipeline : g_gatherPipeline;
        GfxRenderTargetLayout &cachedLayout = composite ? g_sceneTargetLayout : g_gatherTargetLayout;
        if (pipeline != nullptr && cachedLayout.key == layout.key)
            return true;

        if (pipeline != nullptr) {
            wgpuRenderPipelineRelease(pipeline);
            pipeline = nullptr;
        }

        if (!build_pipeline(layout, composite, pipeline))
            return false;

        cachedLayout = layout;
        return true;
    }

    // Render worker thread: fullscreen DoF gather.
    void on_draw(
        ModContext *, const GfxDrawContext *ctx, const void *payload, size_t payloadSize, void *)
    {
        if (payloadSize != sizeof(DrawPayload))
            return;

        DrawPayload data;
        std::memcpy(&data, payload, sizeof(data));
        if (data.color == nullptr || (!data.composite && data.depth == nullptr) ||
            (!data.composite && ctx->uniform_buffer == nullptr) ||
            !ensure_pipeline(ctx->layout, data.composite != 0))
        {
            return;
        }

        WGPUBindGroupEntry entries[4] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
                                         WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
        uint32_t entryCount = 0;
        if (data.composite) {
            entries[0].binding = 1;
            entries[0].textureView = data.color;
            entries[1].binding = 3;
            entries[1].sampler = g_sampler;
            entryCount = 2;
        }
        else {
            entries[0].binding = 0;
            entries[0].buffer = ctx->uniform_buffer;
            entries[0].offset = data.uniform_offset;
            entries[0].size = data.uniform_size;
            entries[1].binding = 1;
            entries[1].textureView = data.color;
            entries[2].binding = 2;
            entries[2].textureView = data.depth;
            entries[3].binding = 3;
            entries[3].sampler = g_sampler;
            entryCount = 4;
        }
        WGPUBindGroupDescriptor bindGroupDesc = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bindGroupDesc.layout = data.composite ? g_compositeBgl : g_bgl;
        bindGroupDesc.entryCount = entryCount;
        bindGroupDesc.entries = entries;
        WGPUBindGroup bindGroup = wgpuDeviceCreateBindGroup(ctx->device, &bindGroupDesc);
        if (bindGroup == nullptr) {
            return;
        }

        wgpuRenderPassEncoderSetPipeline(
            ctx->pass, data.composite ? g_compositePipeline : g_gatherPipeline);
        wgpuRenderPassEncoderSetBindGroup(ctx->pass, 0, bindGroup, 0, nullptr);
        wgpuRenderPassEncoderDraw(ctx->pass, 3, 1, 0, 0);
        wgpuBindGroupRelease(bindGroup);
    }

    // ---------------------------------------------------------------------------------------------
    // Game-side parameter computation (mirrors the focus logic in drawDepth2)
    // ---------------------------------------------------------------------------------------------
    float compute_focus_dist(const view_class *view)
    {
        float focusDist = (view->lookat.center - view->lookat.eye).abs();

        if (dCam_getBody()->Mode() == 4 || dCam_getBody()->Mode() == 7) {
            return focusDist;
        }

        camera_class *camera_p = (camera_class *)dComIfGp_getCamera(0);
        int cam_id = dComIfGp_getPlayerCameraID(0);
        camera_process_class *camProc = dComIfGp_getCamera(cam_id);
        float fovScale = 60.0f / (camProc != nullptr ? fopCamM_GetFovy(camProc) : 48.0f);
        dAttention_c *attention = dComIfGp_getAttention();

        if (attention->LockonTruth()) {
            fopAc_ac_c *atn_actor =
                fopAcM_SearchByID(daPy_getLinkPlayerActorClass()->getAtnActorID());
            if (atn_actor != nullptr && camera_p != nullptr)
            {
                focusDist = atn_actor->current.pos.abs(camera_p->view.lookat.eye);
            }
        }
        else if (dComIfGp_event_runCheck() && fovScale < 3.0f &&
                 g_env_light.field_0x126c < 999999.0f)
        {
            focusDist = g_env_light.field_0x126c;
        }
        return focusDist;
    }

    static float normalize_dof_strength(float alpha, bool hasAttention)
    {
        float normalizedStrength = 0.0f; // [0, 1] for the effect strength
        if (hasAttention) {
            normalizedStrength = (alpha + 254.0f) / 509.0f; // alpha is in [-254, 255]
        }
        else {
            normalizedStrength = (alpha + 255.0f) / 75.0f; // alpha is in [-255, -180]
        }
        return std::clamp(normalizedStrength, 0.0f, 1.0f);
    }

    static bool build_dof_params(const view_class *view, const view_port_class *port, DofParams &params, float &normalizedStrength)
    {
        // Same final value drawDepth2 computes for l_tevColor0.a.
        float alpha = g_env_light.field_0x1264;
        const float ambientFarBlur =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarAmbientFarBlur, 20), 0, 100)) / 100.0f;
        const float ambientFarDistance =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarAmbientFarDistance, 8000), 0, 20000));
        const bool ambientEnabled = get_bool_option(g_cvarAmbientEnabled, true);
        const bool hasGameFocus = (alpha > -254.0f);
        if (!hasGameFocus && (!ambientEnabled || ambientFarBlur <= 0.0f))
            return false; // disable the effect (no gather pass)

        const float attentionPoint = g_env_light.mDemoAttentionPoint;
        const bool hasAttention = (0.0f != attentionPoint);
        const float intensity = static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarIntensity, 60), 0, 200)) / 100.0f;
        const float focusStrength = hasGameFocus ? (normalize_dof_strength(alpha, hasAttention) * intensity) : 0.0f;
        const float ambientStrength = (ambientEnabled && !hasAttention) ? (ambientFarBlur * intensity) : 0.0f;
        normalizedStrength = std::max(focusStrength, ambientStrength);

        // Skip the gather pass if the effect is very weak, to avoid unnecessary GPU work.
        if (normalizedStrength < 0.02f)
            return false;

        const float focusDist =
            std::max(compute_focus_dist(view), 1.0f);
        const float focusPct =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarFocusRange, 40), 0, 200)) / 100.0f;
        const float farPct =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarFarFalloff, 200), 10, 1000)) / 100.0f;
        const float nearPct =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarNearFalloff, 50), 10, 1000)) / 100.0f;
        const float minRange =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarMinFocusRange, 800), 0, 2000));
        const float zoom =
            std::clamp(60.0f / view->fovy, 0.5f, 3.0f);
        const float maxBlurFrac =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarMaxBlur, 10), 1, 60)) / 1000.0f;
        const float tapCount =
            static_cast<float>(std::clamp<int64_t>(get_int_option(g_cvarTapCount, 16), 8, 64));

        params = {};
        params.maxRadiusFrac = maxBlurFrac * zoom;
        params.strength = focusStrength;
        params.focusDist = focusDist;
        params.focusRange = std::max(focusDist * focusPct, minRange);
        params.farFalloff = focusDist * farPct;
        params.nearFalloff = focusDist * nearPct;
        params.nearZ = view->near_;
        params.farZ = view->far_;
        params.reversedZ = g_deviceInfo.uses_reversed_z ? 1.0f : 0.0f;
        params.nearBlur = (attentionPoint < 0.0f) ? 1.0f : 0.0f;
        params.tapCount = tapCount;
        params.ambientFarDistance = ambientFarDistance;
        params.ambientStrength = ambientStrength;
        params.rect[0] = port->x_orig / FB_WIDTH;
        params.rect[1] = port->y_orig / FB_HEIGHT;
        params.rect[2] = (port->x_orig + port->width) / FB_WIDTH;
        params.rect[3] = (port->y_orig + port->height) / FB_HEIGHT;

        return true;
    }

    // Runs after the original drawDepth2 (which has already updated g_env_light.field_0x1264).
    void on_draw_depth2_post(ModContext *, void *args, void *, void *)
    {
        if (!get_bool_option(g_cvarEnabled, false) || daPy_getLinkPlayerActorClass() == nullptr)
            return;

        auto *view = mods::arg<view_class *>(args, 0);
        auto *port = mods::arg<view_port_class *>(args, 1);
        if (view == nullptr || port == nullptr)
            return;

        DofParams params{};
        float strength = 0.0f;
        if (!build_dof_params(view, port, params, strength))
            return;

        GfxResolveDesc resolveDesc = GFX_RESOLVE_DESC_INIT;
        resolveDesc.color = true;
        resolveDesc.depth = true;
        GfxResolvedTargets resolved = GFX_RESOLVED_TARGETS_INIT;
        if (svc_gfx->resolve_pass(mod_ctx, &resolveDesc, &resolved) != MOD_OK ||
            resolved.color == nullptr || resolved.depth == nullptr)
            return;

        const uint32_t divisor = get_bool_option(g_cvarHalfResolution, false) ? 2u : 1u;
        const uint32_t gatherWidth = std::max((resolved.width + divisor - 1u) / divisor, 1u);
        const uint32_t gatherHeight = std::max((resolved.height + divisor - 1u) / divisor, 1u);
        if (svc_gfx->create_pass(mod_ctx, gatherWidth, gatherHeight) != MOD_OK)
            return;

        GfxRange uniformRange{0, 0};
        if (svc_gfx->push_uniform(mod_ctx, &params, sizeof(params), &uniformRange) != MOD_OK) {
            GfxResolveDesc closeDesc = GFX_RESOLVE_DESC_INIT;
            closeDesc.color = false;
            svc_gfx->resolve_pass(mod_ctx, &closeDesc, &resolved);
            return;
        }
        const DrawPayload gatherPayload{
            resolved.color, resolved.depth, uniformRange.offset, uniformRange.size, 0};
        if (svc_gfx->push_draw(mod_ctx, g_drawType, &gatherPayload, sizeof(gatherPayload)) != MOD_OK) {
            GfxResolveDesc closeDesc = GFX_RESOLVE_DESC_INIT;
            closeDesc.color = false;
            svc_gfx->resolve_pass(mod_ctx, &closeDesc, &resolved);
            return;
        }

        GfxResolveDesc gatherResolveDesc = GFX_RESOLVE_DESC_INIT;
        gatherResolveDesc.depth = false;
        GfxResolvedTargets gathered = GFX_RESOLVED_TARGETS_INIT;
        if (svc_gfx->resolve_pass(mod_ctx, &gatherResolveDesc, &gathered) != MOD_OK ||
            gathered.color == nullptr)
            return;

        const DrawPayload compositePayload{gathered.color, nullptr, 0, 0, 1};
        svc_gfx->push_draw(mod_ctx, g_drawType, &compositePayload, sizeof(compositePayload));
    }

    DEFINE_HOOK_SYMBOL("drawDepth2", void(view_class *, view_port_class *, int), DrawDepth2);

    // ---------------------------------------------------------------------------------------------
    // UI
    // ---------------------------------------------------------------------------------------------
    void add_control(UiElementHandle pane, const UiControlDesc &desc) {
        svc_ui->pane_add_control(mod_ctx, pane, &desc, nullptr);
    }

    void add_number(UiElementHandle pane, const char *label, ConfigVarHandle cvar, int64_t min,
                    int64_t max, int64_t step, const char *suffix, const char *help)
    {
        UiControlDesc control = UI_CONTROL_DESC_INIT;
        control.kind = UI_CONTROL_NUMBER;
        control.label = label;
        control.help_rml = help;
        control.binding = UI_BINDING_CONFIG_VAR;
        control.config_var = cvar;
        control.min = min;
        control.max = max;
        control.step = step;
        control.suffix = suffix;
        add_control(pane, control);
    }

    void add_toggle(UiElementHandle pane, const char* label, ConfigVarHandle cvar, const char* help) {
        UiControlDesc control = UI_CONTROL_DESC_INIT;
        control.kind = UI_CONTROL_TOGGLE;
        control.label = label;
        control.help_rml = help;
        control.binding = UI_BINDING_CONFIG_VAR;
        control.config_var = cvar;
        add_control(pane, control);
    }

    ModResult build_controls_tab(ModContext *, UiWindowHandle, UiElementHandle left, UiElementHandle right, void *, ModError *)
    {
        (void)right;

        add_toggle(left, "Enabled", g_cvarEnabled,
                   "Enable the Depth of Field effect.");

        svc_ui->pane_add_section(mod_ctx, left, "Look");
        add_number(left, "Intensity", g_cvarIntensity, 0, 200, 5, "%",
                   "Overall multiplier on the DoF effect strength.");
        add_number(left, "Max Blur", g_cvarMaxBlur, 1, 60, 1, " /1000 of height",
                   "Blur radius at full defocus, as a fraction of screen height. 10 is about 11 px at 1080p.");

        svc_ui->pane_add_section(mod_ctx, left, "Ambient Blur");
        add_toggle(left, "Enabled", g_cvarAmbientEnabled,
                   "Enable a subtle DoF blur effect on distant objects.");
        add_number(left, "Ambient Blur Intensity", g_cvarAmbientFarBlur, 0, 100, 5, "%",
                   "Intensity of the ambient DoF effect.");
        add_number(left, "Ambient Blur Distance", g_cvarAmbientFarDistance, 0, 20000, 500, " units",
                   "World-space units where ambient DoF begins.");

        svc_ui->pane_add_section(mod_ctx, left, "Focus");
        add_number(left, "Focus Range", g_cvarFocusRange, 0, 200, 5, "%",
                   "Depth of the sharp zone, as a percentage of the focus distance.");
        add_number(left, "Min Focus Range", g_cvarMinFocusRange, 0, 2000, 25, " units",
                   "Floor for the sharp zone so close focus targets keep their own limbs sharp.");
        add_number(left, "Far Falloff", g_cvarFarFalloff, 10, 1000, 10, "%",
                   "Distance past the sharp zone to reach full blur (behind the focus).");
        add_number(left, "Near Falloff", g_cvarNearFalloff, 10, 1000, 10, "%",
                   "Same for in front of the focus; only used when near blur is active (some cutscenes).");

        svc_ui->pane_add_section(mod_ctx, left, "Performance");
        add_toggle(left, "Half Resolution", g_cvarHalfResolution,
                   "Render the DoF gather pass at half the scene resolution for better performance.");
        add_number(left, "Tap Count", g_cvarTapCount, 8, 64, 8, " taps",
                   "Number of samples to use in the gather pass.");
        return MOD_OK;
    }

    void on_controls_window_closed(ModContext *, UiWindowHandle, void *)
    {
        g_controlsWindow = 0;
    }

    void on_open_controls(ModContext *, void *)
    {
        if (g_controlsWindow != 0)
            return;

        UiTabDesc tabs[1] = {UI_TAB_DESC_INIT};
        tabs[0].title = "Controls";
        tabs[0].build = build_controls_tab;
        UiWindowDesc desc = UI_WINDOW_DESC_INIT;
        desc.tabs = tabs;
        desc.tab_count = 1;
        desc.on_closed = on_controls_window_closed;
        if (svc_ui->window_push(mod_ctx, &desc, &g_controlsWindow) != MOD_OK) {
            svc_log->error(mod_ctx, "failed to open depth of field controls window");
        }
    }

    ModResult build_panel(ModContext *, UiElementHandle panel, void *, ModError *)
    {
        add_toggle(panel, "Enabled", g_cvarEnabled, "Enable the Depth of Field effect.");
        UiControlDesc control = UI_CONTROL_DESC_INIT;
        control.kind = UI_CONTROL_BUTTON;
        control.label = "Open Controls";
        control.on_pressed = on_open_controls;
        add_control(panel, control);
        return MOD_OK;
    }

} // namespace

extern "C"
{

    MOD_EXPORT ModResult mod_initialize(ModError *error)
    {
        ModResult result = register_bool_option("effectEnabled", true, g_cvarEnabled, error);
        if (result != MOD_OK)
            return result;

        if ((result = register_int_option("intensity", 60, g_cvarIntensity, error)) != MOD_OK ||
            (result = register_bool_option("ambientBlurEnabled", true, g_cvarAmbientEnabled, error)) != MOD_OK ||
            (result = register_int_option("ambientFarBlur", 20, g_cvarAmbientFarBlur, error)) != MOD_OK ||
            (result = register_int_option("ambientFarDistance", 8000, g_cvarAmbientFarDistance, error)) != MOD_OK ||
            (result = register_int_option("maxBlur", 10, g_cvarMaxBlur, error)) != MOD_OK ||
            (result = register_int_option("focusRange", 40, g_cvarFocusRange, error)) != MOD_OK ||
            (result = register_int_option("minFocusRange", 800, g_cvarMinFocusRange, error)) != MOD_OK ||
            (result = register_int_option("farFalloff", 200, g_cvarFarFalloff, error)) != MOD_OK ||
            (result = register_int_option("nearFalloff", 50, g_cvarNearFalloff, error)) != MOD_OK ||
            (result = register_int_option("tapCount", 16, g_cvarTapCount, error)) != MOD_OK ||
            (result = register_bool_option("halfResolution", false, g_cvarHalfResolution, error)) != MOD_OK)
        {
            return result;
        }

        result = svc_resource->load(mod_ctx, "dof_hq.wgsl", &g_shaderSource);
        if (result != MOD_OK || g_shaderSource.data == nullptr) {
            return mods::set_error(error, result, "failed to load dof_hq.wgsl");
        }

        if (svc_gfx->get_device_info(mod_ctx, &g_deviceInfo) != MOD_OK) {
            return mods::set_error(error, MOD_ERROR, "failed to query device info");
        }

        if (!init_gpu()) {
            release_gpu();
            return mods::set_error(error, MOD_ERROR, "failed to create depth of field GPU resources");
        }

        // The shader module keeps its own copy of the source.
        svc_resource->free(mod_ctx, &g_shaderSource);

        GfxDrawTypeDesc drawDesc = GFX_DRAW_TYPE_DESC_INIT;
        drawDesc.label = "dof_hq";
        drawDesc.draw = on_draw;
        if (svc_gfx->register_draw_type(mod_ctx, &drawDesc, &g_drawType) != MOD_OK) {
            release_gpu();
            return mods::set_error(error, MOD_ERROR, "failed to register draw type");
        }

        if (mods::hook::add_post<DrawDepth2>(on_draw_depth2_post) != MOD_OK) {
            release_gpu();
            return mods::set_error(error, MOD_UNAVAILABLE, "failed to hook drawDepth2");
        }

        UiModsPanelDesc panelDesc = UI_MODS_PANEL_DESC_INIT;
        panelDesc.build = build_panel;
        svc_ui->register_mods_panel(mod_ctx, &panelDesc);
        return MOD_OK;
    }

    MOD_EXPORT ModResult mod_update(ModError *)
    {
        return MOD_OK;
    }

    MOD_EXPORT ModResult mod_shutdown(ModError *)
    {
        release_gpu();
        return MOD_OK;
    }
}
