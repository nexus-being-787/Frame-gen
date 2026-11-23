#pragma once

#include <getopt.h>

#include <atomic>
#include <optional>
#include <string_view>
#include <utility>

#include "gamescope_shared.h"

extern const char *gamescope_optstring;
extern const struct option *gamescope_options;

extern std::atomic< bool > g_bRun;

extern int g_nNestedWidth;
extern int g_nNestedHeight;
extern int g_nNestedRefresh; // mHz
extern int g_nNestedUnfocusedRefresh; // mHz
extern int g_nNestedDisplayIndex;

extern uint32_t g_nOutputWidth;
extern uint32_t g_nOutputHeight;
extern bool g_bForceRelativeMouse;
extern int g_nOutputRefresh; // mHz
extern bool g_bOutputHDREnabled;
extern bool g_bForceInternal;

extern bool g_bForceCompositionRotation;
extern uint32_t g_uOutputRotation;

extern bool g_bFullscreen;

extern bool g_bGrabbed;

extern float g_mouseSensitivity;
extern const char *g_sOutputName;

enum class GamescopeUpscaleFilter : uint32_t
{
    LINEAR = 0,
    NEAREST,
    FSR,
    NIS,
    PIXEL,
    SGSR,

    FROM_VIEW = 0xF, // internal
};

static constexpr bool UpscaleFilterUsesSharpness( GamescopeUpscaleFilter eFilter )
{
    return eFilter == GamescopeUpscaleFilter::FSR ||
           eFilter == GamescopeUpscaleFilter::NIS ||
           eFilter == GamescopeUpscaleFilter::SGSR;
}

// cs_sgsr reads the plain sampler slot, not the YCbCr one, and thresholds in 8-bit SDR units.
static constexpr bool SgsrSupportsInput( GamescopeAppTextureColorspace eColorspace, bool bYcbcr )
{
    return !bYcbcr && ( eColorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_LINEAR || eColorspace == GAMESCOPE_APP_TEXTURE_COLORSPACE_SRGB );
}

// Sharp ran FSR before SGSR existed, so HDR keeps that rather than losing the sharpening. Neither pre-pass reads the YCbCr slot.
static constexpr GamescopeUpscaleFilter ResolveUpscaleFilter( GamescopeUpscaleFilter eFilter, GamescopeAppTextureColorspace eColorspace, bool bYcbcr )
{
    if ( eFilter != GamescopeUpscaleFilter::SGSR || SgsrSupportsInput( eColorspace, bYcbcr ) )
        return eFilter;
    return bYcbcr ? GamescopeUpscaleFilter::LINEAR : GamescopeUpscaleFilter::FSR;
}

static constexpr bool DoesHardwareSupportUpscaleFilter( GamescopeUpscaleFilter eFilter )
{
    // Could do nearest someday... AMDGPU DC supports custom tap placement to an extent.

    return eFilter == GamescopeUpscaleFilter::LINEAR;
}

enum class GamescopeUpscaleScaler : uint32_t
{
    AUTO,
    INTEGER,
    FIT,
    FILL,
    STRETCH,
};

struct UpscaleSettings_t
{
    GamescopeUpscaleFilter eFilter{};
    GamescopeUpscaleScaler eScaler{};
    int nSharpness{};
};

// XXX(misyl): This is bad! We shouldnt change the upscaler like this at all!!!
// We should move this to business logic in paint_window or something!
static constexpr UpscaleSettings_t GetUpscaleSettings(
    bool bFocusIsSteam,
    GamescopeUpscaleFilter eWantedFilter,
    GamescopeUpscaleScaler eWantedScaler,
    int nWantedSharpness )
{
    if ( bFocusIsSteam )
        return UpscaleSettings_t{ GamescopeUpscaleFilter::LINEAR, GamescopeUpscaleScaler::FIT, nWantedSharpness };

    return UpscaleSettings_t{ eWantedFilter, eWantedScaler, nWantedSharpness };
}

// One name table for the --filter option and the scaling_filter command, so the two cannot drift.
inline std::optional<GamescopeUpscaleFilter> ParseUpscaleFilter( std::string_view svName )
{
    static constexpr std::pair<std::string_view, GamescopeUpscaleFilter> k_Filters[] =
    {
        { "linear",  GamescopeUpscaleFilter::LINEAR },
        { "nearest", GamescopeUpscaleFilter::NEAREST },
        { "fsr",     GamescopeUpscaleFilter::FSR },
        { "nis",     GamescopeUpscaleFilter::NIS },
        { "pixel",   GamescopeUpscaleFilter::PIXEL },
        { "sgsr",    GamescopeUpscaleFilter::SGSR },
    };
    for ( const auto &[svFilterName, eFilter] : k_Filters )
    {
        if ( svFilterName == svName )
            return eFilter;
    }
    return std::nullopt;
}

extern GamescopeUpscaleFilter g_wantedUpscaleFilter;
extern GamescopeUpscaleScaler g_wantedUpscaleScaler;
extern int g_upscaleFilterSharpness;

enum class GamescopeFrameGenMode : uint32_t
{
    OFF = 0,
    DUPLICATE = 1,
    BLEND = 2,
    OPTICAL_FLOW = 3,
};

inline std::optional<GamescopeFrameGenMode> ParseFrameGenMode( std::string_view svName )
{
    static constexpr std::pair<std::string_view, GamescopeFrameGenMode> k_Modes[] =
    {
        { "off",         GamescopeFrameGenMode::OFF },
        { "duplicate",   GamescopeFrameGenMode::DUPLICATE },
        { "dup",         GamescopeFrameGenMode::DUPLICATE },
        { "blend",       GamescopeFrameGenMode::BLEND },
        { "opticalflow", GamescopeFrameGenMode::OPTICAL_FLOW },
        { "flow",        GamescopeFrameGenMode::OPTICAL_FLOW },
    };
    for ( const auto &[svModeName, eMode] : k_Modes )
    {
        if ( svModeName == svName )
            return eMode;
    }
    return std::nullopt;
}

inline const char *FrameGenModeName( GamescopeFrameGenMode eMode )
{
    switch ( eMode )
    {
        case GamescopeFrameGenMode::DUPLICATE:    return "duplicate";
        case GamescopeFrameGenMode::BLEND:        return "blend";
        case GamescopeFrameGenMode::OPTICAL_FLOW: return "opticalflow";
        case GamescopeFrameGenMode::OFF:
        default:                                  return "off";
    }
}

enum class GamescopeFrameGenDebug : uint32_t
{
    OFF = 0,
    SPLIT = 1,
    FLOW = 2,
    CONFIDENCE = 3,
    OCCLUSION = 4,
};

inline std::optional<GamescopeFrameGenDebug> ParseFrameGenDebug( std::string_view svName )
{
    static constexpr std::pair<std::string_view, GamescopeFrameGenDebug> k_DebugModes[] =
    {
        { "off",        GamescopeFrameGenDebug::OFF },
        { "split",      GamescopeFrameGenDebug::SPLIT },
        { "flow",       GamescopeFrameGenDebug::FLOW },
        { "confidence", GamescopeFrameGenDebug::CONFIDENCE },
        { "occlusion",  GamescopeFrameGenDebug::OCCLUSION },
    };
    for ( const auto &[svDebugName, eDebug] : k_DebugModes )
    {
        if ( svDebugName == svName )
            return eDebug;
    }
    return std::nullopt;
}

inline const char *FrameGenDebugName( GamescopeFrameGenDebug eDebug )
{
    switch ( eDebug )
    {
        case GamescopeFrameGenDebug::SPLIT:      return "split";
        case GamescopeFrameGenDebug::FLOW:       return "flow";
        case GamescopeFrameGenDebug::CONFIDENCE: return "confidence";
        case GamescopeFrameGenDebug::OCCLUSION:  return "occlusion";
        case GamescopeFrameGenDebug::OFF:
        default:                                 return "off";
    }
}

extern std::atomic<GamescopeFrameGenMode> g_eFrameGenMode;
extern std::atomic<GamescopeFrameGenDebug> g_eFrameGenDebug;
extern std::atomic<int> g_nFrameGenMultiplier;
extern std::atomic<int> g_nFrameGenFlowDiv;
extern float g_flFrameGenSceneCut;
extern float g_flFrameGenFlowScale;
extern float g_flFrameGenSharpness;
extern bool g_bFrameGenStats;
extern bool g_bFrameGenLowLatency;
extern bool g_bFrameGenAdaptive;
extern bool g_bFrameGenAutoBudget;
extern float g_flFrameGenMask[4]; // normalized x,y,w,h static HUD lock rect
extern const char *g_sFrameGenCsvPath;

void gamescope_cycle_framegen_mode();
void gamescope_toggle_framegen_flow_dup();
void gamescope_cycle_framegen_debug();
void gamescope_cycle_framegen_multiplier();
void gamescope_cycle_framegen_quality();

extern bool g_bBorderlessOutputWindow;

extern bool g_bExposeWayland;

extern bool g_bRt;

extern int g_nXWaylandCount;
extern bool g_bNoTouchPointerEmulation;

extern uint32_t g_preferVendorID;
extern uint32_t g_preferDeviceID;

