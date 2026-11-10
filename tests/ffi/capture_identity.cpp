#include "capture_identity.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

#include "preparation_fixture.h"

namespace {
    constexpr std::uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;

    void hash_byte(std::uint64_t& hash, std::uint8_t value) {
        hash ^= value;
        hash *= kFnvPrime;
    }

    void mix(std::uint64_t& hash, std::uint64_t value) {
        hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    }

    void mix_key(std::uint64_t& hash, const std::string& value) {
        mix(hash, value.size());
        if (!value.empty()) {
            std::uint64_t bytes = kFnvOffset;
            for (const char c : value) {
                hash_byte(bytes, static_cast<std::uint8_t>(c));
            }
            mix(hash, bytes);
        }
    }

    void mix_canonical_float(std::uint64_t& hash, float value) {
        mix(hash, std::bit_cast<std::uint32_t>(value == 0.0f ? 0.0f : value));
    }

    template <typename T>
    void mix_finite_float(std::uint64_t& hash, T value) {
        if (std::isfinite(value)) {
            mix(hash, std::bit_cast<std::uint32_t>(static_cast<float>(value)));
        }
    }

    template <typename T>
    void mix_finite_canonical_float(std::uint64_t& hash, T value) {
        if (std::isfinite(value)) {
            mix_canonical_float(hash, static_cast<float>(value));
        }
    }

    template <unsigned ByteCount>
    void hash_little_endian(std::uint64_t& hash, std::uint64_t value) {
        static_assert(ByteCount <= sizeof(value));
        for (unsigned i = 0; i < ByteCount; ++i) {
            hash_byte(hash, static_cast<std::uint8_t>((value >> (i * 8)) & 0xffu));
        }
    }

    std::uint64_t hash_diffusion(const FjDiffusionParams& value) {
        std::uint64_t hash = kFnvOffset;
        for (const char c : std::string_view("diffusion-authored")) {
            hash_byte(hash, static_cast<std::uint8_t>(c));
        }
        hash_byte(hash, 0);
        hash_little_endian<4>(hash, 1);
        hash_byte(hash, value.active != 0 ? 1 : 0);
        hash_byte(hash, static_cast<std::uint8_t>(value.family));
        for (double scalar : {value.strength, value.spatial_scale, value.halo_warmth, value.core_intensity, value.core_size, value.halo_intensity, value.halo_size, value.bloom_intensity, value.bloom_size}) {
            if (scalar == 0.0) {
                scalar = 0.0;
            }
            hash_little_endian<8>(hash, std::bit_cast<std::uint64_t>(scalar));
        }
        return hash;
    }
} // namespace

namespace JuicerTestSupport {
    std::uint64_t hash_input_identity(const PreparationInput& input) {
        const auto& p = input.values;
        if (p.scan_route > 3) {
            throw std::out_of_range("invalid captured scan route");
        }
        std::uint64_t hash = 0;
        mix_key(hash, input.film_key);
        mix(hash, static_cast<std::uint8_t>(p.scan_route));
        if (p.scan_route == 1 || p.scan_route == 3) {
            mix_key(hash, input.print_key);
            mix(hash, static_cast<std::uint64_t>(p.enl_ill));
            mix_finite_float(hash, p.print_exposure);
            mix_finite_float(hash, p.print_preflash_exposure);
            mix(hash, std::bit_cast<std::uint64_t>(p.print_gamma_factor));
            mix(hash, static_cast<std::uint64_t>(p.normalize_print_exposure));
            mix(hash, static_cast<std::uint64_t>(p.print_exposure_compensation));
            for (double value : p.print_ui_ymc_cc)
                mix_finite_canonical_float(hash, value);
            mix_finite_canonical_float(hash, p.preflash_m_filter_cc);
            mix_finite_canonical_float(hash, p.preflash_y_filter_cc);
        }
        mix(hash, static_cast<std::uint64_t>(p.spectral_upsampling_mode));
        mix(hash, static_cast<std::uint64_t>(p.ref_ill));
        mix(hash, static_cast<std::uint64_t>(p.print_shadow_compensation_factor * 10000.0));
        mix(hash, static_cast<std::uint64_t>(p.print_shadow_compensation_density * 10000.0));
        mix(hash, static_cast<std::uint64_t>(p.print_shadow_compensation_transition * 10000.0));
        mix(hash, p.glare_active != 0 ? 1 : 0);
        mix_finite_float(hash, p.glare_percent);
        mix_finite_float(hash, p.glare_roughness);
        mix_finite_float(hash, p.glare_blur_sigma_px);
        const auto& dir = p.dir_couplers;
        mix(hash, dir.active != 0 ? 1 : 0);
        mix_canonical_float(hash, dir.amount);
        mix_canonical_float(hash, dir.inhibition_same_layer);
        mix_canonical_float(hash, dir.inhibition_interlayer);
        mix_canonical_float(hash, dir.diffusion_size_um);
        mix_canonical_float(hash, dir.diffusion_tail_um);
        mix_canonical_float(hash, dir.diffusion_tail_weight);
        mix(hash, dir.gamma_use_stock != 0 ? 1 : 0);
        for (float value : dir.gamma_same_layer_rgb)
            mix_canonical_float(hash, value);
        for (float value : dir.gamma_interlayer_r_to_gb)
            mix_canonical_float(hash, value);
        for (float value : dir.gamma_interlayer_g_to_rb)
            mix_canonical_float(hash, value);
        for (float value : dir.gamma_interlayer_b_to_rg)
            mix_canonical_float(hash, value);
        for (float value : dir.langmuir_donor_k_rgb)
            mix_canonical_float(hash, value);
        for (float value : dir.langmuir_receiver_k_rgb)
            mix_canonical_float(hash, value);
        mix(hash, static_cast<std::uint64_t>(p.scanner_lut_resolution));
        mix(hash, static_cast<std::uint64_t>(p.input_color_space));
        mix(hash, static_cast<std::uint64_t>(p.input_cctf_decoding));
        mix(hash, static_cast<std::uint64_t>(p.output_color_space));
        mix(hash, static_cast<std::uint64_t>(p.output_cctf_encoding));
        mix(hash, static_cast<std::uint64_t>(p.output_gamut_compression_enabled));
        if (p.spectral_upsampling_mode == 0) {
            mix(hash, static_cast<std::uint64_t>(p.hanatos2025_adaptation_window));
            mix(hash, static_cast<std::uint64_t>(p.hanatos2025_adaptation_surface));
        }
        if (p.spectral_upsampling_mode == 0 || p.spectral_upsampling_mode == 2) {
            mix(hash, static_cast<std::uint64_t>(p.input_compression_enabled));
        }
        mix(hash, p.camera_filter_override != 0 ? 1 : 0);
        if (p.camera_filter_override != 0) {
            for (double value : p.camera_filter_uv)
                mix_finite_float(hash, value);
            for (double value : p.camera_filter_ir)
                mix_finite_float(hash, value);
        }
        mix(hash, static_cast<std::uint64_t>(p.camera_auto_exposure_enabled));
        mix(hash, static_cast<std::uint64_t>(p.camera_metering_method));
        mix_finite_float(hash, p.camera_exposure_compensation_ev);
        mix_canonical_float(hash, p.camera_film_format_long_edge_mm);
        mix_canonical_float(hash, p.film_gamma_factor);
        mix(hash, p.scatter_halation.active != 0 ? 1 : 0);
        if (p.scatter_halation.active != 0) {
            mix_canonical_float(hash, p.scatter_halation.scatter_amount);
            mix_canonical_float(hash, p.scatter_halation.scatter_spatial_scale);
            mix_canonical_float(hash, p.scatter_halation.halation_amount);
            mix_canonical_float(hash, p.scatter_halation.halation_spatial_scale);
        }
        mix_finite_float(hash, p.scanner_lens_blur_sigma_px);
        for (double value : p.scanner_unsharp_mask)
            mix_finite_float(hash, value);
        mix(hash, static_cast<std::uint64_t>(p.scanner_black_correction));
        mix(hash, static_cast<std::uint64_t>(p.scanner_white_correction));
        mix_finite_float(hash, p.scanner_black_level);
        mix_finite_float(hash, p.scanner_white_level);
        mix(hash, 0);
        mix(hash, hash_diffusion(p.camera_diffusion));
        mix(hash, 3);
        mix(hash, hash_diffusion(p.enlarger_diffusion));
        const auto& grain = p.grain;
        if (grain.active != 0) {
            mix(hash, 1);
            mix(hash, grain.sublayers_active != 0 ? 1 : 0);
            mix_finite_float(hash, grain.particle_area_um2);
            for (float value : grain.particle_scale_cmy)
                mix_finite_float(hash, value);
            for (float value : grain.particle_scale_layers)
                mix_finite_float(hash, value);
            for (float value : grain.particle_density_min_cmy)
                mix_finite_float(hash, value);
            for (float value : grain.uniformity_cmy)
                mix_finite_float(hash, value);
            mix_finite_float(hash, grain.correlation_sigma_px);
            mix_finite_float(hash, grain.dye_cloud_blur_um);
            for (float value : grain.micro_structure)
                mix_finite_float(hash, value);
            mix(hash, static_cast<std::uint64_t>(grain.sublayer_count));
            mix_finite_float(hash, grain.amplitude);
            mix_finite_float(hash, grain.chroma_shared_weight);
            mix_finite_float(hash, grain.chroma_independent_weight);
            mix_finite_float(hash, grain.coarse_weight);
            mix_finite_float(hash, grain.mid_weight);
            mix_finite_float(hash, grain.size_mix_scale);
            mix_finite_float(hash, grain.clump_temporal_mix);
            mix_finite_float(hash, grain.clump_morph_period_sec);
            mix(hash, static_cast<std::uint64_t>(grain.debug_view));
        }
        const auto positive = [](auto value) {
            return std::isfinite(value) && value > 0;
        };
        if (positive(p.film_dust_amount) || positive(p.film_scratch_amount) ||
            positive(p.gate_dust_amount) || positive(p.gate_scratch_amount) || positive(p.gate_weave_amount)) {
            mix(hash, 1);
            mix_finite_canonical_float(hash, p.film_dust_amount);
            mix_finite_canonical_float(hash, p.film_scratch_amount);
            mix_finite_canonical_float(hash, p.gate_dust_amount);
            mix_finite_canonical_float(hash, p.gate_scratch_amount);
            if (std::isfinite(p.gate_weave_amount)) {
                mix(hash, std::bit_cast<std::uint64_t>(p.gate_weave_amount));
            }
        }
        return hash;
    }
} // namespace JuicerTestSupport
