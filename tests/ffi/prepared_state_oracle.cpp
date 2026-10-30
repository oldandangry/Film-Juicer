#include "prepared_state_oracle.h"

#include <array>
#include <bit>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "juicer_legacy_api.h"

namespace PreparedStateOracle {
    namespace {

        template <class T>
        std::string hex_word(T value) {
            using U = std::conditional_t<sizeof(T) == 4, std::uint32_t, std::uint64_t>;
            U bits{};
            if constexpr (std::is_floating_point_v<T>)
                bits = std::bit_cast<U>(value);
            else
                bits = static_cast<U>(value);
            std::ostringstream out;
            out << std::hex << std::nouppercase << std::setfill('0') << std::setw(sizeof(T) * 2) << bits;
            return out.str();
        }

        template <class T>
        Json scalar(T value) {
            if constexpr (std::is_same_v<T, float>)
                return {{"type", "f32"}, {"bits", hex_word(value)}};
            else if constexpr (std::is_same_v<T, double>)
                return {{"type", "f64"}, {"bits", hex_word(value)}};
            else if constexpr (std::is_same_v<T, std::uint32_t>)
                return {{"type", "u32"}, {"value", value}};
            else if constexpr (std::is_same_v<T, std::int32_t>)
                return {{"type", "i32"}, {"value", value}};
            else if constexpr (std::is_same_v<T, std::uint64_t>)
                return {{"type", "u64"}, {"value", value}};
            else if constexpr (std::is_same_v<T, std::int64_t>)
                return {{"type", "i64"}, {"value", value}};
            else
                static_assert(!sizeof(T), "unsupported oracle scalar type");
        }

        template <class T, std::size_t N>
        Json array(const std::string& source, const T (&values)[N], std::vector<std::size_t> shape, const char* order) {
            Json words = Json::array();
            for (const auto& value : values)
                words.push_back(scalar(value).at(std::is_floating_point_v<T> ? "bits" : "value"));
            std::size_t count = 1;
            for (auto extent : shape)
                count *= extent;
            return {{"source", source}, {"element_type", scalar(T{}).at("type")}, {"shape", shape}, {"count", count}, {"order", order}, {"words", std::move(words)}};
        }

        template <class T, std::size_t R, std::size_t C>
        Json array(const std::string& source, const T (&values)[R][C], std::vector<std::size_t> shape, const char* order) {
            Json words = Json::array();
            for (const auto& row : values)
                for (const auto& value : row)
                    words.push_back(scalar(value).at(std::is_floating_point_v<T> ? "bits" : "value"));
            std::size_t count = 1;
            for (auto extent : shape)
                count *= extent;
            return {{"source", source}, {"element_type", scalar(T{}).at("type")}, {"shape", shape}, {"count", count}, {"order", order}, {"words", std::move(words)}};
        }

        Json span(const std::string& source, FjFloatSpan value, const char* order) {
            Json words = Json::array();
            for (std::size_t i = 0; i < value.count; ++i)
                words.push_back(hex_word(value.data[i]));
            return {{"source", source}, {"element_type", "f32"}, {"shape", {value.count}}, {"count", value.count}, {"order", order}, {"words", std::move(words)}};
        }

        Json string_view(FjStringView value) {
            return {{"type", "utf8"}, {"value", std::string(value.data ? value.data : "", value.count)}};
        }

        Json span_node(const std::string& source, FjFloatSpan value, const char* order) {
            auto node = span(source, value, order);
            if (value.count == 0)
                return node;
            std::vector<std::size_t> shape;
            if (source.ends_with("tc_lut_rgba"))
                shape = {192, 192, 4};
            else if (source.ends_with("hanatos_surface_rgb"))
                shape = {3, 15};
            else if (source.ends_with("sensitivity_rgb") || source.ends_with("mallett_basis") || source.ends_with("film_density_cmy") || source.ends_with("sensitivity_cmy"))
                shape = {81, 3};
            else if (source.ends_with("density_rgb") || source.ends_with("density_cmy")) {
                if (value.count % 3 == 0)
                    shape = {value.count / 3, 3};
            } else if (source.ends_with("xy")) {
                if (value.count % 2 == 0)
                    shape = {value.count / 2, 2};
            } else if (source.ends_with("cmax") || source.ends_with("gamut_cmax")) {
                if (value.count == std::size_t{64} * 720u)
                    shape = {64u, 720u};
            }
            if (!shape.empty())
                node["shape"] = shape;
            return node;
        }

        Json span_array_node(const std::string& source, Json elements, std::vector<std::size_t> shape, const char* order) {
            std::size_t count = 1;
            for (const auto extent : shape)
                count *= extent;
            return {{"source", source}, {"element_type", "span<f32>"}, {"shape", std::move(shape)}, {"count", count}, {"order", order}, {"elements", std::move(elements)}};
        }

        auto bound_diffusion_params(const FjDiffusionParams& value) {
            const auto& [active, family, strength, spatial_scale, halo_warmth, core_intensity, core_size, halo_intensity, halo_size, bloom_intensity, bloom_size] = value;
            return std::tie(active, family, strength, spatial_scale, halo_warmth, core_intensity, core_size, halo_intensity, halo_size, bloom_intensity, bloom_size);
        }

        auto bound_dir_params(const FjFilmDirParams& value) {
            const auto& [active, amount, inhibition_same_layer, inhibition_interlayer, diffusion_size_um, diffusion_tail_um, diffusion_tail_weight, gamma_use_stock, gamma_same_layer_rgb, gamma_interlayer_r_to_gb, gamma_interlayer_g_to_rb, gamma_interlayer_b_to_rg, langmuir_donor_k_rgb, langmuir_receiver_k_rgb] = value;
            return std::tie(active, amount, inhibition_same_layer, inhibition_interlayer, diffusion_size_um, diffusion_tail_um, diffusion_tail_weight, gamma_use_stock, gamma_same_layer_rgb, gamma_interlayer_r_to_gb, gamma_interlayer_g_to_rb, gamma_interlayer_b_to_rg, langmuir_donor_k_rgb, langmuir_receiver_k_rgb);
        }

        auto bound_grain_params(const FjVisualGrainParams& value) {
            const auto& [active, sublayers_active, particle_area_um2, particle_scale_cmy, particle_scale_layers, particle_density_min_cmy, uniformity_cmy, correlation_sigma_px, dye_cloud_blur_um, micro_structure, sublayer_count, amplitude, chroma_mix, chroma_shared_weight, chroma_independent_weight, coarse_weight, mid_weight, size_mix_scale, clump_temporal_mix, clump_morph_period_sec, debug_view] = value;
            return std::tie(active, sublayers_active, particle_area_um2, particle_scale_cmy, particle_scale_layers, particle_density_min_cmy, uniformity_cmy, correlation_sigma_px, dye_cloud_blur_um, micro_structure, sublayer_count, amplitude, chroma_mix, chroma_shared_weight, chroma_independent_weight, coarse_weight, mid_weight, size_mix_scale, clump_temporal_mix, clump_morph_period_sec, debug_view);
        }

        auto bound_halation_params(const FjPreparationHalation& value) {
            const auto& [active, scatter_amount, scatter_spatial_scale, halation_amount, halation_spatial_scale] = value;
            return std::tie(active, scatter_amount, scatter_spatial_scale, halation_amount, halation_spatial_scale);
        }

        auto bound_preparation_params(const FjPreparationParams& value) {
            const auto& [print_exposure, print_preflash_exposure, print_gamma_factor, preflash_m_filter_cc, preflash_y_filter_cc, print_shadow_compensation_factor, print_shadow_compensation_density, print_shadow_compensation_transition, glare_percent, glare_roughness, glare_blur_sigma_px, camera_exposure_compensation_ev, camera_film_format_long_edge_mm, film_gamma_factor, scanner_lens_blur_sigma_px, scanner_black_level, scanner_white_level, gate_weave_amount, scanner_unsharp_mask, print_ui_ymc_cc, camera_filter_uv, camera_filter_ir, camera_diffusion, enlarger_diffusion, dir_couplers, film_profile_key, print_profile_key, spectral_upsampling_mode, ref_ill, enl_ill, normalize_print_exposure, print_exposure_compensation, input_color_space, input_cctf_decoding, input_compression_enabled, hanatos2025_adaptation_window, hanatos2025_adaptation_surface, camera_auto_exposure_enabled, camera_metering_method, scatter_halation, scanner_black_correction, scanner_white_correction, scanner_lut_resolution, output_color_space, output_cctf_encoding, output_gamut_compression_enabled, grain, film_dust_amount, film_scratch_amount, gate_dust_amount, gate_scratch_amount, scan_route, glare_active, camera_filter_override] = value;
            return std::tie(print_exposure, print_preflash_exposure, print_gamma_factor, preflash_m_filter_cc, preflash_y_filter_cc, print_shadow_compensation_factor, print_shadow_compensation_density, print_shadow_compensation_transition, glare_percent, glare_roughness, glare_blur_sigma_px, camera_exposure_compensation_ev, camera_film_format_long_edge_mm, film_gamma_factor, scanner_lens_blur_sigma_px, scanner_black_level, scanner_white_level, gate_weave_amount, scanner_unsharp_mask, print_ui_ymc_cc, camera_filter_uv, camera_filter_ir, camera_diffusion, enlarger_diffusion, dir_couplers, film_profile_key, print_profile_key, spectral_upsampling_mode, ref_ill, enl_ill, normalize_print_exposure, print_exposure_compensation, input_color_space, input_cctf_decoding, input_compression_enabled, hanatos2025_adaptation_window, hanatos2025_adaptation_surface, camera_auto_exposure_enabled, camera_metering_method, scatter_halation, scanner_black_correction, scanner_white_correction, scanner_lut_resolution, output_color_space, output_cctf_encoding, output_gamut_compression_enabled, grain, film_dust_amount, film_scratch_amount, gate_dust_amount, gate_scratch_amount, scan_route, glare_active, camera_filter_override);
        }

        Json diffusion(const std::string& p, const FjDiffusionRecipe& v) {
            return {{"domain", scalar(v.domain)}, {"active", scalar(v.active)}, {"family", scalar(v.family)}, {"scatter_fraction", scalar(v.scatter_fraction)}, {"group_weights", array(p + ".group_weights", v.group_weights, {3}, "group")}, {"group_lambda_um", array(p + ".group_lambda_um", v.group_lambda_um, {3}, "group")}, {"warmth", scalar(v.warmth)}, {"spatial_scale", scalar(v.spatial_scale)}, {"resolved_hash", scalar(v.resolved_hash)}, {"hash", scalar(v.hash)}};
        }

        Json dust(const std::string&, const FjDust& v) {
#define F(name)               \
    {                         \
        #name, scalar(v.name) \
    }
            return {F(cell_width_mm), F(cell_height_mm), F(slot_probability), F(softness_min_mm), F(softness_max_mm), F(softness_size_cap_fraction), F(support_x_mm), F(support_y_mm), F(fiber_fraction), F(fiber_drift_fraction), F(fiber_first_knot_min), F(fiber_first_knot_max), F(fiber_second_knot_min), F(fiber_second_knot_max), F(fiber_interior_width_min_fraction), F(fiber_interior_width_max_fraction), F(diameter_min_mm), F(diameter_bulk_max_mm), F(diameter_max_mm), F(diameter_tail_fraction), F(fiber_length_min_mm), F(fiber_length_max_mm), F(fiber_width_min_mm), F(fiber_width_max_mm), F(opacity_faint_cumulative), F(opacity_intermediate_cumulative), F(compact_opacity_min), F(compact_opacity_faint_end), F(compact_opacity_intermediate_end), F(compact_opacity_max), F(fiber_opacity_min), F(fiber_opacity_faint_end), F(fiber_opacity_intermediate_end), F(fiber_opacity_max), F(compact_dominant_aspect_min), F(compact_dominant_aspect_max), F(compact_subsidiary_scale_min), F(compact_subsidiary_scale_max), F(compact_subsidiary_aspect_min), F(compact_subsidiary_aspect_max), F(compact_subsidiary_offset_max), F(compact_subsidiary_angle_max_radians)};
#undef F
        }

        Json scratch(const std::string&, const FjScratch& v) {
#define F(name)               \
    {                         \
        #name, scalar(v.name) \
    }
            return {F(cell_width_mm), F(cell_height_mm), F(slot_probability), F(softness_min_mm), F(softness_max_mm), F(softness_size_cap_fraction), F(support_x_mm), F(support_y_mm), F(length_min_mm), F(length_bulk_max_mm), F(length_max_mm), F(length_tail_fraction), F(width_min_mm), F(width_bulk_max_mm), F(width_max_mm), F(width_tail_fraction), F(drift_fraction), F(first_knot_min), F(first_knot_max), F(second_knot_min), F(second_knot_max), F(interior_width_min_fraction), F(interior_width_max_fraction), F(interior_depth_min_fraction), F(interior_depth_max_fraction), F(endpoint_abrupt_probability), F(interruption_probability), F(gap_center_min), F(gap_center_max), F(gap_span_min), F(gap_span_max), F(scuff_probability), F(scuff_length_max_mm), F(scuff_angle_max_radians), F(strength_min), F(strength_max)};
#undef F
        }

        Json bounds(const std::string& p, const FjFilmDensityBounds& v) {
            return {{"route", scalar(v.route)}, {"medium", scalar(v.medium)}, {"polarity", scalar(v.polarity)}, {"source", scalar(v.source)}, {"minimum_cmy", array(p + ".minimum_cmy", v.minimum_cmy, {3}, "CMY")}, {"maximum_cmy", array(p + ".maximum_cmy", v.maximum_cmy, {3}, "CMY")}, {"inv_span_cmy", array(p + ".inv_span_cmy", v.inv_span_cmy, {3}, "CMY")}, {"authored_minimum_cmy", array(p + ".authored_minimum_cmy", v.authored_minimum_cmy, {3}, "CMY")}, {"authored_maximum_cmy", array(p + ".authored_maximum_cmy", v.authored_maximum_cmy, {3}, "CMY")}, {"hash", scalar(v.hash)}};
        }

        Json spectral(const std::string& p, const FjPreparedSpectralView& v) {
            Json o{{"lambda_nm", span_node(p + ".lambda_nm", v.lambda_nm, "wavelength")}, {"illuminant", span_node(p + ".illuminant", v.illuminant, "wavelength")}, {"baseline_min", span_node(p + ".baseline_min", v.baseline_min, "wavelength")}, {"baseline_mid", span_node(p + ".baseline_mid", v.baseline_mid, "wavelength")}, {"inv_yn", scalar(v.inv_yn)}, {"white_xyz", array(p + ".white_xyz", v.white_xyz, {3}, "XYZ")}, {"has_baseline", scalar(v.has_baseline)}, {"illuminant_hash", scalar(v.illuminant_hash)}, {"hash", scalar(v.hash)}};
            Json observer = Json::array(), weighted = Json::array(), dyes = Json::array();
            for (std::size_t i = 0; i < 3; ++i) {
                const char* channel = i == 0 ? "X" : i == 1 ? "Y"
                                                            : "Z";
                observer.push_back(span_node(p + ".observer_xyz[" + std::to_string(i) + "]", v.observer_xyz[i], channel));
                weighted.push_back(span_node(p + ".weighted_xyz[" + std::to_string(i) + "]", v.weighted_xyz[i], channel));
                const char* dye = i == 0 ? "C" : i == 1 ? "M"
                                                        : "Y";
                dyes.push_back(span_node(p + ".dyes_cmy[" + std::to_string(i) + "]", v.dyes_cmy[i], dye));
            }
            o["observer_xyz"] = span_array_node(p + ".observer_xyz", std::move(observer), {3}, "X/Y/Z");
            o["weighted_xyz"] = span_array_node(p + ".weighted_xyz", std::move(weighted), {3}, "X/Y/Z");
            o["dyes_cmy"] = span_array_node(p + ".dyes_cmy", std::move(dyes), {3}, "C/M/Y");
            return o;
        }

        Json development(const std::string& p, const FjFilmDevelopment& v) {
            Json layers = Json::array();
            for (std::size_t l = 0; l < 3; ++l) {
                for (std::size_t c = 0; c < 3; ++c)
                    layers.push_back(span_node(p + ".density_layers[" + std::to_string(l) + "][" + std::to_string(c) + "]", v.density_layers[l][c], "layer/channel/sample"));
            }
            auto density = span_node(p + ".density_rgb", v.density_rgb, "sample/RGB");
            if (v.density_rgb.count != 0)
                density["shape"] = std::vector<std::size_t>{v.log_exposure.count, 3};
            return {{"log_exposure", span_node(p + ".log_exposure", v.log_exposure, "sample")}, {"density_rgb", std::move(density)}, {"density_layers", span_array_node(p + ".density_layers", std::move(layers), {3, 3}, "layer/channel")}, {"gamma_rgb", array(p + ".gamma_rgb", v.gamma_rgb, {3}, "RGB")}, {"density_curves_hash", scalar(v.density_curves_hash)}, {"density_layers_hash", scalar(v.density_layers_hash)}, {"hash", scalar(v.hash)}};
        }

        Json exposure(const std::string& p, const FjFilmExposure& v) {
            return {{"input_color_space", scalar(v.input_color_space)}, {"method", scalar(v.method)}, {"flags", scalar(v.flags)}, {"manual_exposure_ev", scalar(v.manual_exposure_ev)}, {"route_correction_scale", scalar(v.route_correction_scale)}, {"mallett_green_midgray_scale", scalar(v.mallett_green_midgray_scale)}, {"input_rgb_to_xyz", array(p + ".input_rgb_to_xyz", v.input_rgb_to_xyz, {3, 3}, "row-major")}, {"input_xyz_adapt", array(p + ".input_xyz_adapt", v.input_xyz_adapt, {3, 3}, "row-major")}, {"xyz_to_linear_srgb", array(p + ".xyz_to_linear_srgb", v.xyz_to_linear_srgb, {3, 3}, "row-major")}, {"sensitivity_rgb", span_node(p + ".sensitivity_rgb", v.sensitivity_rgb, "wavelength/RGB")}, {"tc_lut_rgba", span_node(p + ".tc_lut_rgba", v.tc_lut_rgba, "tc0/tc1/RGBA")}, {"mallett_illuminant", span_node(p + ".mallett_illuminant", v.mallett_illuminant, "wavelength")}, {"mallett_basis", span_node(p + ".mallett_basis", v.mallett_basis, "wavelength/basis-RGB")}, {"sensitivity_hash", scalar(v.sensitivity_hash)}, {"tc_lut_hash", scalar(v.tc_lut_hash)}, {"hash", scalar(v.hash)}};
        }

        Json dir_view(const std::string& p, const FjFilmDirView& v) {
            Json axes = Json::array();
            for (std::size_t i = 0; i < 3; ++i)
                axes.push_back(span_node(p + ".compensated_axes_rgb[" + std::to_string(i) + "]", v.compensated_axes_rgb[i], "RGB/sample"));
            return {{"polarity", scalar(v.polarity)}, {"mode", scalar(v.mode)}, {"matrix_rgb", array(p + ".matrix_rgb", v.matrix_rgb, {3, 3}, "row-major RGB")}, {"diffusion_size_um", scalar(v.diffusion_size_um)}, {"diffusion_tail_um", scalar(v.diffusion_tail_um)}, {"diffusion_tail_weight", scalar(v.diffusion_tail_weight)}, {"density_max_rgb", array(p + ".density_max_rgb", v.density_max_rgb, {3}, "RGB")}, {"density_ref_rgb", array(p + ".density_ref_rgb", v.density_ref_rgb, {3}, "RGB")}, {"donor_k_rgb", array(p + ".donor_k_rgb", v.donor_k_rgb, {3}, "RGB")}, {"receiver_c_ref_rgb", array(p + ".receiver_c_ref_rgb", v.receiver_c_ref_rgb, {3}, "RGB")}, {"receiver_kr_rgb", array(p + ".receiver_kr_rgb", v.receiver_kr_rgb, {3}, "RGB")}, {"compensated_axes_rgb", span_array_node(p + ".compensated_axes_rgb", std::move(axes), {3}, "RGB")}, {"compensated_axes_hash", scalar(v.compensated_axes_hash)}, {"hash", scalar(v.hash)}};
        }

        Json output_encoding(const std::string& p, const FjOutputEncoding& v) {
            return {{"schema", scalar(v.schema)}, {"cctf_kind", scalar(v.cctf_kind)}, {"gamma", scalar(v.gamma)}, {"a", scalar(v.a)}, {"b", scalar(v.b)}, {"c", scalar(v.c)}, {"d", scalar(v.d)}, {"linear_cutoff", scalar(v.linear_cutoff)}, {"dwg_to_output", array(p + ".dwg_to_output", v.dwg_to_output, {3, 3}, "row-major")}};
        }

        Json scanner_color(const std::string& p, const FjScannerColor& v) {
            return {{"cat02", array(p + ".cat02", v.cat02, {3, 3}, "row-major")}, {"xyz_to_rgb", array(p + ".xyz_to_rgb", v.xyz_to_rgb, {3, 3}, "row-major")}, {"illuminant_xyz", array(p + ".illuminant_xyz", v.illuminant_xyz, {3}, "XYZ")}, {"output_space", scalar(v.output_space)}, {"flags", scalar(v.flags)}, {"encoding", output_encoding(p + ".encoding", v.encoding)}, {"active_gamut_recipe_hash", scalar(v.active_gamut_recipe_hash)}, {"hash", scalar(v.hash)}};
        }

        Json grain(const std::string& p, const FjVisualGrainRecipe& v) {
            return {{"active", scalar(v.active)}, {"sublayers_active", scalar(v.sublayers_active)}, {"axis_finite_cmy", array(p + ".axis_finite_cmy", v.axis_finite_cmy, {3}, "CMY")}, {"axis_block_prefix_max_cmy", array(p + ".axis_block_prefix_max_cmy", v.axis_block_prefix_max_cmy, {3, 16}, "CMY/axis-block")}, {"particle_area_um2", scalar(v.particle_area_um2)}, {"particle_scale_cmy", array(p + ".particle_scale_cmy", v.particle_scale_cmy, {3}, "CMY")}, {"particle_scale_layers", array(p + ".particle_scale_layers", v.particle_scale_layers, {3}, "layer")}, {"particle_density_min_cmy", array(p + ".particle_density_min_cmy", v.particle_density_min_cmy, {3}, "CMY")}, {"uniformity_cmy", array(p + ".uniformity_cmy", v.uniformity_cmy, {3}, "CMY")}, {"correlation_sigma_px", scalar(v.correlation_sigma_px)}, {"dye_cloud_blur_um", scalar(v.dye_cloud_blur_um)}, {"micro_structure", array(p + ".micro_structure", v.micro_structure, {2}, "fine/coarse")}, {"sublayer_count", scalar(v.sublayer_count)}, {"amplitude", scalar(v.amplitude)}, {"chroma_mix", scalar(v.chroma_mix)}, {"chroma_shared_weight", scalar(v.chroma_shared_weight)}, {"chroma_independent_weight", scalar(v.chroma_independent_weight)}, {"fine_weight", scalar(v.fine_weight)}, {"mid_weight", scalar(v.mid_weight)}, {"coarse_weight", scalar(v.coarse_weight)}, {"size_mix_scale", scalar(v.size_mix_scale)}, {"clump_temporal_mix", scalar(v.clump_temporal_mix)}, {"clump_morph_period_sec", scalar(v.clump_morph_period_sec)}, {"debug_view", scalar(v.debug_view)}, {"layers_hash", scalar(v.layers_hash)}, {"hash", scalar(v.hash)}};
        }

        Json effects(const std::string& p, const FjPhysicalEffectsRecipe& v) {
            return {{"film_dust", dust(p + ".film_dust", v.film_dust)}, {"film_scratch", scratch(p + ".film_scratch", v.film_scratch)}, {"gate_dust", dust(p + ".gate_dust", v.gate_dust)}, {"gate_scratch", scratch(p + ".gate_scratch", v.gate_scratch)}, {"gate_weave_amount", scalar(v.gate_weave_amount)}, {"active", scalar(v.active)}, {"hash", scalar(v.hash)}};
        }

        Json print_value(const std::string& p, const FjPrint& v) {
            auto density = span_node(p + ".density_cmy", v.density_cmy, "sample/CMY");
            if (v.density_cmy.count != 0)
                density["shape"] = std::vector<std::size_t>{v.log_exposure.count, 3};
            return {{"film_density_cmy", span_node(p + ".film_density_cmy", v.film_density_cmy, "wavelength/CMY")}, {"film_base_density", span_node(p + ".film_base_density", v.film_base_density, "wavelength")}, {"sensitivity_cmy", span_node(p + ".sensitivity_cmy", v.sensitivity_cmy, "wavelength/CMY")}, {"log_exposure", span_node(p + ".log_exposure", v.log_exposure, "sample")}, {"density_cmy", std::move(density)}, {"main_illuminant", span_node(p + ".main_illuminant", v.main_illuminant, "wavelength")}, {"preflash_illuminant", span_node(p + ".preflash_illuminant", v.preflash_illuminant, "wavelength")}, {"flags", scalar(v.flags)}, {"main_cc_cmy", array(p + ".main_cc_cmy", v.main_cc_cmy, {3}, "CMY")}, {"preflash_cc_cmy", array(p + ".preflash_cc_cmy", v.preflash_cc_cmy, {3}, "CMY")}, {"exposure", scalar(v.exposure)}, {"preflash_exposure", scalar(v.preflash_exposure)}, {"preflash_raw_cmy", array(p + ".preflash_raw_cmy", v.preflash_raw_cmy, {3}, "CMY")}, {"normalizer", scalar(v.normalizer)}, {"film_density_hash", scalar(v.film_density_hash)}, {"profile_tables_hash", scalar(v.profile_tables_hash)}, {"main_illuminant_hash", scalar(v.main_illuminant_hash)}, {"preflash_illuminant_hash", scalar(v.preflash_illuminant_hash)}, {"preflash_raw_hash", scalar(v.preflash_raw_hash)}, {"balance_hash", scalar(v.balance_hash)}, {"hash", scalar(v.hash)}};
        }

        Json params_diffusion(const std::string&, const FjDiffusionParams& v) {
            (void)bound_diffusion_params(v);
            return {{"active", scalar(v.active)}, {"family", scalar(v.family)}, {"strength", scalar(v.strength)}, {"spatial_scale", scalar(v.spatial_scale)}, {"halo_warmth", scalar(v.halo_warmth)}, {"core_intensity", scalar(v.core_intensity)}, {"core_size", scalar(v.core_size)}, {"halo_intensity", scalar(v.halo_intensity)}, {"halo_size", scalar(v.halo_size)}, {"bloom_intensity", scalar(v.bloom_intensity)}, {"bloom_size", scalar(v.bloom_size)}};
        }

        Json params_dir(const std::string& p, const FjFilmDirParams& v) {
            (void)bound_dir_params(v);
            return {{"active", scalar(v.active)}, {"amount", scalar(v.amount)}, {"inhibition_same_layer", scalar(v.inhibition_same_layer)}, {"inhibition_interlayer", scalar(v.inhibition_interlayer)}, {"diffusion_size_um", scalar(v.diffusion_size_um)}, {"diffusion_tail_um", scalar(v.diffusion_tail_um)}, {"diffusion_tail_weight", scalar(v.diffusion_tail_weight)}, {"gamma_use_stock", scalar(v.gamma_use_stock)}, {"gamma_same_layer_rgb", array(p + ".gamma_same_layer_rgb", v.gamma_same_layer_rgb, {3}, "RGB")}, {"gamma_interlayer_r_to_gb", array(p + ".gamma_interlayer_r_to_gb", v.gamma_interlayer_r_to_gb, {2}, "G/B")}, {"gamma_interlayer_g_to_rb", array(p + ".gamma_interlayer_g_to_rb", v.gamma_interlayer_g_to_rb, {2}, "R/B")}, {"gamma_interlayer_b_to_rg", array(p + ".gamma_interlayer_b_to_rg", v.gamma_interlayer_b_to_rg, {2}, "R/G")}, {"langmuir_donor_k_rgb", array(p + ".langmuir_donor_k_rgb", v.langmuir_donor_k_rgb, {3}, "RGB")}, {"langmuir_receiver_k_rgb", array(p + ".langmuir_receiver_k_rgb", v.langmuir_receiver_k_rgb, {3}, "RGB")}};
        }

        Json params_grain(const std::string& p, const FjVisualGrainParams& v) {
            (void)bound_grain_params(v);
            return {{"active", scalar(v.active)}, {"sublayers_active", scalar(v.sublayers_active)}, {"particle_area_um2", scalar(v.particle_area_um2)}, {"particle_scale_cmy", array(p + ".particle_scale_cmy", v.particle_scale_cmy, {3}, "CMY")}, {"particle_scale_layers", array(p + ".particle_scale_layers", v.particle_scale_layers, {3}, "layer")}, {"particle_density_min_cmy", array(p + ".particle_density_min_cmy", v.particle_density_min_cmy, {3}, "CMY")}, {"uniformity_cmy", array(p + ".uniformity_cmy", v.uniformity_cmy, {3}, "CMY")}, {"correlation_sigma_px", scalar(v.correlation_sigma_px)}, {"dye_cloud_blur_um", scalar(v.dye_cloud_blur_um)}, {"micro_structure", array(p + ".micro_structure", v.micro_structure, {2}, "fine/coarse")}, {"sublayer_count", scalar(v.sublayer_count)}, {"amplitude", scalar(v.amplitude)}, {"chroma_mix", scalar(v.chroma_mix)}, {"chroma_shared_weight", scalar(v.chroma_shared_weight)}, {"chroma_independent_weight", scalar(v.chroma_independent_weight)}, {"coarse_weight", scalar(v.coarse_weight)}, {"mid_weight", scalar(v.mid_weight)}, {"size_mix_scale", scalar(v.size_mix_scale)}, {"clump_temporal_mix", scalar(v.clump_temporal_mix)}, {"clump_morph_period_sec", scalar(v.clump_morph_period_sec)}, {"debug_view", scalar(v.debug_view)}};
        }

        Json recipe_optics(const std::string& p, const FjSpatialOpticsRecipe& v) {
            const auto& h = v.scatter_halation;
            return {{"camera_diffusion", diffusion(p + ".camera_diffusion", v.camera_diffusion)}, {"camera_lens_blur", {{"sigma_um", scalar(v.camera_lens_blur.sigma_um)}, {"hash", scalar(v.camera_lens_blur.hash)}}}, {"scatter_halation", {{"scatter_active", scalar(h.scatter_active)}, {"back_reflection_active", scalar(h.back_reflection_active)}, {"scatter_amount", scalar(h.scatter_amount)}, {"scatter_spatial_scale", scalar(h.scatter_spatial_scale)}, {"halation_spatial_scale", scalar(h.halation_spatial_scale)}, {"first_sigma_um", array(p + ".scatter_halation.first_sigma_um", h.first_sigma_um, {3}, "CMY")}, {"total_strength", array(p + ".scatter_halation.total_strength", h.total_strength, {3}, "CMY")}, {"hash", scalar(h.hash)}}}, {"enlarger_diffusion", diffusion(p + ".enlarger_diffusion", v.enlarger_diffusion)}, {"hash", scalar(v.hash)}};
        }

        void check_node(const Json& node, const std::string& path) {
            if (!node.is_object())
                throw std::runtime_error(path + ": expected object node");
            if (node.contains("type")) {
                const auto type = node.at("type").get<std::string>();
                if (type == "live-backing") {
                    if (node.size() != 2 || !node.contains("relation") || !node.at("relation").is_string())
                        throw std::runtime_error(path + ": invalid live-backing node");
                    const auto relation = node.at("relation").get<std::string>();
                    if (relation != "input_hull.entry_identity" && relation != "output_boundary.entry_identity")
                        throw std::runtime_error(path + ": unsupported backing relation");
                    return;
                }
                if (type == "utf8") {
                    if (node.size() != 2 || !node.contains("value") || !node.at("value").is_string())
                        throw std::runtime_error(path + ": invalid utf8 node");
                } else if (type == "f32" || type == "f64") {
                    const auto width = type == "f32" ? 8U : 16U;
                    if (node.size() != 2 || !node.contains("bits") || !node.at("bits").is_string() || node.at("bits").get<std::string>().size() != width)
                        throw std::runtime_error(path + ": invalid " + type + " bit width");
                    for (const char c : node.at("bits").get<std::string>())
                        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                            throw std::runtime_error(path + ": invalid hexadecimal bits");
                } else if (type == "u32" || type == "i32" || type == "u64" || type == "i64") {
                    if (node.size() != 2 || !node.contains("value") || !node.at("value").is_number_integer())
                        throw std::runtime_error(path + ": invalid integer node");
                    if (type == "u32" && (!node.at("value").is_number_unsigned() || node.at("value").get<std::uint64_t>() > UINT32_MAX))
                        throw std::runtime_error(path + ": value exceeds u32");
                    if (type == "u64" && !node.at("value").is_number_unsigned())
                        throw std::runtime_error(path + ": value is not u64");
                    if (type == "i32") {
                        const auto signed_value = node.at("value").get<std::int64_t>();
                        if (signed_value < INT32_MIN || signed_value > INT32_MAX)
                            throw std::runtime_error(path + ": value exceeds i32");
                    }
                    if (type == "i64" && node.at("value").is_number_unsigned())
                        throw std::runtime_error(path + ": value is not i64");
                    const auto value = type == "u64" ? std::int64_t{} : node.at("value").get<std::int64_t>();
                    const auto in_range = [value](std::int64_t maximum) {
                        return value >= 0 && value <= maximum;
                    };
                    if ((path == "expected.route" || path == "actual.route" || path.ends_with(".scanner_lut.route") || path.ends_with(".scanner_correction.route") || path.ends_with(".scanner_post_effects.route") || path.ends_with(".scanner_bounds.route") || path.ends_with(".enlarger_film_bounds.route")) && !in_range(3))
                        throw std::runtime_error(path + ": unsupported route tag");
                    if ((path == "expected.polarity" || path == "actual.polarity" || path.ends_with(".scanner_lut.capture_polarity")) && !in_range(1))
                        throw std::runtime_error(path + ": unsupported polarity tag");
                    if (path.ends_with(".dir_recipe.mode") && !in_range(2))
                        throw std::runtime_error(path + ": unsupported DIR mode tag");
                    if (path.ends_with(".calibration") && !in_range(2))
                        throw std::runtime_error(path + ": unsupported calibration tag");
                    if (path.ends_with(".encoding.cctf_kind") && !in_range(5))
                        throw std::runtime_error(path + ": unsupported CCTF tag");
                    if (path.ends_with(".print_normalization_mode") && !in_range(3))
                        throw std::runtime_error(path + ": unsupported print normalization tag");
                    if (path.ends_with(".film_exposure.method") && !in_range(2))
                        throw std::runtime_error(path + ": unsupported spectral upsampling tag");
                    if (path.ends_with(".film_exposure.input_color_space") && !in_range(3))
                        throw std::runtime_error(path + ": unsupported input color-space tag");
                    if ((path.ends_with(".scanner_color.output_space") || path.ends_with(".output_color.color_space")) && !in_range(8))
                        throw std::runtime_error(path + ": unsupported output color-space tag");
                    if (path.ends_with(".input_policy.meter") && !in_range(6))
                        throw std::runtime_error(path + ": unsupported metering tag");
                    if ((path == "expected.raw_midgray_bgr" || path == "actual.raw_midgray_bgr") && !in_range(1))
                        throw std::runtime_error(path + ": unsupported channel-order tag");
                } else
                    throw std::runtime_error(path + ": unsupported scalar type " + type);
                return;
            }
            if (node.contains("source") && node.at("source").is_string()) {
                static const std::array<const char*, 5> common_keys{"source", "element_type", "shape", "count", "order"};
                if (node.size() != 6)
                    throw std::runtime_error(path + ": array node has missing or unknown keys");
                for (const auto* key : common_keys)
                    if (!node.contains(key))
                        throw std::runtime_error(path + ": array node missing " + key);
                if (node.contains("words") == node.contains("elements"))
                    throw std::runtime_error(path + ": array node must have exactly one words/elements member");
                const auto type = node.at("element_type").get<std::string>();
                if (!node.at("shape").is_array() || !node.at("source").is_string() || !node.at("order").is_string())
                    throw std::runtime_error(path + ": invalid array shape/source/order");
                std::size_t count = 1;
                for (const auto& extent : node.at("shape")) {
                    if (!extent.is_number_unsigned())
                        throw std::runtime_error(path + ": invalid array extent");
                    const auto dimension = extent.get<std::size_t>();
                    if (dimension != 0 && count > std::numeric_limits<std::size_t>::max() / dimension)
                        throw std::runtime_error(path + ": array shape overflows size_t");
                    count *= dimension;
                }
                if (node.at("shape").empty())
                    count = 0;
                if (!node.at("count").is_number_unsigned() || count != node.at("count").get<std::size_t>())
                    throw std::runtime_error(path + ": array shape/count mismatch");
                if (type == "span<f32>") {
                    if (node.size() != 6 || !node.contains("elements") || !node.at("elements").is_array() || count != node.at("elements").size())
                        throw std::runtime_error(path + ": span-array shape/count/elements mismatch");
                    for (std::size_t i = 0; i < node.at("elements").size(); ++i)
                        check_node(node.at("elements")[i], path + ".elements[" + std::to_string(i) + "]");
                    return;
                }
                const std::size_t width = type == "f32" || type == "u32" || type == "i32" ? 8 : type == "f64" || type == "u64" || type == "i64" ? 16
                                                                                                                                                : 0;
                if (!width || node.size() != 6 || !node.contains("words") || !node.at("words").is_array() || count != node.at("words").size())
                    throw std::runtime_error(path + ": invalid array element type/word length");
                if (type == "f32" || type == "f64")
                    for (const auto& word : node.at("words")) {
                        if (!word.is_string() || word.get<std::string>().size() != width)
                            throw std::runtime_error(path + ": array word has invalid bit width");
                        for (const char c : word.get<std::string>())
                            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
                                throw std::runtime_error(path + ": array word has invalid hexadecimal bits");
                    }
                if (type == "u32" || type == "u64")
                    for (const auto& word : node.at("words")) {
                        if (!word.is_number_unsigned() || (type == "u32" && word.get<std::uint64_t>() > UINT32_MAX))
                            throw std::runtime_error(path + ": integer array word exceeds its unsigned type");
                    }
                if (type == "i32" || type == "i64")
                    for (const auto& word : node.at("words")) {
                        if (!word.is_number_integer())
                            throw std::runtime_error(path + ": invalid signed integer array word");
                        const auto value = word.get<std::int64_t>();
                        if (type == "i32" && (value < INT32_MIN || value > INT32_MAX))
                            throw std::runtime_error(path + ": integer array word exceeds i32");
                    }
                return;
            }
            for (auto it = node.begin(); it != node.end(); ++it)
                check_node(it.value(), path + "." + it.key());
        }

        void compare_nodes(const Json& expected, const Json& actual, const std::string& path) {
            if (expected.is_object() || actual.is_object()) {
                if (!expected.is_object() || !actual.is_object())
                    throw std::runtime_error(path + ": node kind mismatch");
                for (auto it = expected.begin(); it != expected.end(); ++it) {
                    if (!actual.contains(it.key()))
                        throw std::runtime_error(path + "." + it.key() + ": missing actual field");
                    compare_nodes(it.value(), actual.at(it.key()), path + "." + it.key());
                }
                for (auto it = actual.begin(); it != actual.end(); ++it)
                    if (!expected.contains(it.key()))
                        throw std::runtime_error(path + "." + it.key() + ": unknown actual field");
                return;
            }
            if (expected.is_array() || actual.is_array()) {
                if (!expected.is_array() || !actual.is_array())
                    throw std::runtime_error(path + ": array kind mismatch");
                if (expected.size() != actual.size())
                    throw std::runtime_error(path + ": array length mismatch");
                for (std::size_t i = 0; i < expected.size(); ++i)
                    compare_nodes(expected[i], actual[i], path + "[" + std::to_string(i) + "]");
                return;
            }
            if (expected != actual)
                throw std::runtime_error(path + ": value mismatch");
        }

    } // namespace

    Json view(const FjPreparedStateView& v) {
        const std::string p = "view";
        Json in{{"meter", scalar(v.input_policy.meter)}, {"band_pass_active", scalar(v.input_policy.band_pass_active)}, {"apply_hanatos_window", scalar(v.input_policy.apply_hanatos_window)}, {"apply_hanatos_surface", scalar(v.input_policy.apply_hanatos_surface)}, {"projection_illuminant", scalar(v.input_policy.projection_illuminant)}, {"input_compression_active", scalar(v.input_policy.input_compression_active)}, {"film_format_long_edge_mm", scalar(v.input_policy.film_format_long_edge_mm)}, {"uv", array(p + ".input_policy.uv", v.input_policy.uv, {3}, "UV")}, {"ir", array(p + ".input_policy.ir", v.input_policy.ir, {3}, "IR")}, {"hanatos_spectral_blur", scalar(v.input_policy.hanatos_spectral_blur)}, {"nominal_white_xyz", array(p + ".input_policy.nominal_white_xyz", v.input_policy.nominal_white_xyz, {3}, "XYZ")}, {"projection_white_xyz", array(p + ".input_policy.projection_white_xyz", v.input_policy.projection_white_xyz, {3}, "XYZ")}, {"hanatos_window", span_node(p + ".input_policy.hanatos_window", v.input_policy.hanatos_window, "parameter")}, {"hanatos_surface_rgb", span_node(p + ".input_policy.hanatos_surface_rgb", v.input_policy.hanatos_surface_rgb, "RGB-channel-major/15-coefficients")}, {"input_hull_hash", scalar(v.input_policy.input_hull_hash)}, {"tc_source_hash", scalar(v.input_policy.tc_source_hash)}};
        Json scanner_policy{{"black_correction", scalar(v.scanner_policy.black_correction)}, {"white_correction", scalar(v.scanner_policy.white_correction)}, {"glare_active", scalar(v.scanner_policy.glare_active)}, {"black_level", scalar(v.scanner_policy.black_level)}, {"white_level", scalar(v.scanner_policy.white_level)}, {"glare_percent", scalar(v.scanner_policy.glare_percent)}, {"glare_roughness", scalar(v.scanner_policy.glare_roughness)}, {"glare_blur_sigma_px", scalar(v.scanner_policy.glare_blur_sigma_px)}, {"lens_blur_sigma_px", scalar(v.scanner_policy.lens_blur_sigma_px)}, {"unsharp_sigma_px", scalar(v.scanner_policy.unsharp_sigma_px)}, {"unsharp_amount", scalar(v.scanner_policy.unsharp_amount)}};
        Json hull{{"center_xy", array(p + ".input_hull.center_xy", v.input_hull.center_xy, {2}, "xy")}, {"xy", span_node(p + ".input_hull.xy", v.input_hull.xy, "point/xy")}, {"hash", scalar(v.input_hull.hash)}};
        if (v.input_hull.entry_identity)
            hull["entry_identity"] = {{"type", "live-backing"}, {"relation", "input_hull.entry_identity"}};
        Json boundary{{"cmax", span_node(p + ".output_boundary.cmax", v.output_boundary.cmax, "lightness-major[64][720]")}, {"transform_hash", scalar(v.output_boundary.transform_hash)}, {"contract_hash", scalar(v.output_boundary.contract_hash)}, {"hash", scalar(v.output_boundary.hash)}};
        if (v.output_boundary.entry_identity)
            boundary["entry_identity"] = {{"type", "live-backing"}, {"relation", "output_boundary.entry_identity"}};
        Json corr{{"route", scalar(v.scanner_correction.route)}, {"active", scalar(v.scanner_correction.active)}, {"black_correction", scalar(v.scanner_correction.black_correction)}, {"white_correction", scalar(v.scanner_correction.white_correction)}, {"target_black_linear", scalar(v.scanner_correction.target_black_linear)}, {"target_white_linear", scalar(v.scanner_correction.target_white_linear)}, {"reference_black_y", scalar(v.scanner_correction.reference_black_y)}, {"reference_white_y", scalar(v.scanner_correction.reference_white_y)}, {"xyz_slope", scalar(v.scanner_correction.xyz_slope)}, {"xyz_offset", scalar(v.scanner_correction.xyz_offset)}, {"exposure_scale", scalar(v.scanner_correction.exposure_scale)}, {"schema_version", scalar(v.scanner_correction.schema_version)}, {"hash", scalar(v.scanner_correction.hash)}};
        Json post{{"route", scalar(v.scanner_post_effects.route)}, {"glare_active", scalar(v.scanner_post_effects.glare_active)}, {"glare_percent", scalar(v.scanner_post_effects.glare_percent)}, {"glare_roughness", scalar(v.scanner_post_effects.glare_roughness)}, {"glare_blur_sigma_px", scalar(v.scanner_post_effects.glare_blur_sigma_px)}, {"glare_blur_radius", scalar(v.scanner_post_effects.glare_blur_radius)}, {"lens_blur_sigma_px", scalar(v.scanner_post_effects.lens_blur_sigma_px)}, {"lens_blur_radius", scalar(v.scanner_post_effects.lens_blur_radius)}, {"unsharp_sigma_px", scalar(v.scanner_post_effects.unsharp_sigma_px)}, {"unsharp_radius", scalar(v.scanner_post_effects.unsharp_radius)}, {"unsharp_amount", scalar(v.scanner_post_effects.unsharp_amount)}, {"schema", scalar(v.scanner_post_effects.schema)}, {"hash", scalar(v.scanner_post_effects.hash)}};
        Json enc{{"schema", scalar(v.output_color.encoding.schema)}, {"cctf_kind", scalar(v.output_color.encoding.cctf_kind)}, {"gamma", scalar(v.output_color.encoding.gamma)}, {"a", scalar(v.output_color.encoding.a)}, {"b", scalar(v.output_color.encoding.b)}, {"c", scalar(v.output_color.encoding.c)}, {"d", scalar(v.output_color.encoding.d)}, {"linear_cutoff", scalar(v.output_color.encoding.linear_cutoff)}, {"dwg_to_output", array(p + ".output_color.encoding.dwg_to_output", v.output_color.encoding.dwg_to_output, {3, 3}, "row-major")}};
        Json out_color{{"cat02", array(p + ".output_color.cat02", v.output_color.cat02, {3, 3}, "row-major")}, {"xyz_to_rgb", array(p + ".output_color.xyz_to_rgb", v.output_color.xyz_to_rgb, {3, 3}, "row-major")}, {"illuminant_xyz", array(p + ".output_color.illuminant_xyz", v.output_color.illuminant_xyz, {3}, "XYZ")}, {"color_space", scalar(v.output_color.color_space)}, {"flags", scalar(v.output_color.flags)}, {"native_rgb_to_d65_xyz", array(p + ".output_color.native_rgb_to_d65_xyz", v.output_color.native_rgb_to_d65_xyz, {3, 3}, "row-major")}, {"d65_xyz_to_native_rgb", array(p + ".output_color.d65_xyz_to_native_rgb", v.output_color.d65_xyz_to_native_rgb, {3, 3}, "row-major")}, {"lightness_knee", array(p + ".output_color.lightness_knee", v.output_color.lightness_knee, {3}, "threshold/limit/power")}, {"chroma_knee", array(p + ".output_color.chroma_knee", v.output_color.chroma_knee, {3}, "threshold/limit/power")}, {"gamut_cmax", span_node(p + ".output_color.gamut_cmax", v.output_color.gamut_cmax, "lightness-major[64][720]")}, {"gamut_table_hash", scalar(v.output_color.gamut_table_hash)}, {"gamut_recipe_hash", scalar(v.output_color.gamut_recipe_hash)}, {"encoding", std::move(enc)}};
        Json bal{{"baseline_raw_rgb", array(p + ".print_balance.baseline_raw_rgb", v.print_balance.baseline_raw_rgb, {3}, "RGB")}, {"compensated_raw_rgb", array(p + ".print_balance.compensated_raw_rgb", v.print_balance.compensated_raw_rgb, {3}, "RGB")}, {"factor_midgray", scalar(v.print_balance.factor_midgray)}, {"factor_midgray_comp", scalar(v.print_balance.factor_midgray_comp)}, {"normalizer", scalar(v.print_balance.normalizer)}, {"filtered_main_illuminant_hash", scalar(v.print_balance.filtered_main_illuminant_hash)}, {"hash", scalar(v.print_balance.hash)}};
        Json lut{{"route", scalar(v.scanner_lut.route)}, {"medium", scalar(v.scanner_lut.medium)}, {"capture_polarity", scalar(v.scanner_lut.capture_polarity)}, {"resolution", scalar(v.scanner_lut.resolution)}, {"density_bounds_hash", scalar(v.scanner_lut.density_bounds_hash)}, {"channel_density_hash", scalar(v.scanner_lut.channel_density_hash)}, {"base_density_hash", scalar(v.scanner_lut.base_density_hash)}, {"viewing_illuminant_hash", scalar(v.scanner_lut.viewing_illuminant_hash)}, {"observer_hash", scalar(v.scanner_lut.observer_hash)}, {"hash", scalar(v.scanner_lut.hash)}};
        return {{"film_profile_key", string_view(v.film_profile_key)}, {"print_profile_key", string_view(v.print_profile_key)}, {"film_profile_asset_version", scalar(v.film_profile_asset_version)}, {"print_profile_asset_version", scalar(v.print_profile_asset_version)}, {"profile_route_hash", scalar(v.profile_route_hash)}, {"recipe_hash", scalar(v.recipe_hash)}, {"upload_core_hash", scalar(v.upload_core_hash)}, {"scanner_hash", scalar(v.scanner_hash)}, {"route", scalar(v.route)}, {"polarity", scalar(v.polarity)}, {"input_policy", std::move(in)}, {"scanner_policy", std::move(scanner_policy)}, {"film_exposure", exposure(p + ".film_exposure", v.film_exposure)}, {"film_development", development(p + ".film_development", v.film_development)}, {"dir_recipe", dir_view(p + ".dir_recipe", v.dir_recipe)}, {"grain_minimum", {{"cmy", array(p + ".grain_minimum.cmy", v.grain_minimum.cmy, {3}, "CMY")}}}, {"scanner_bounds", bounds(p + ".scanner_bounds", v.scanner_bounds)}, {"enlarger_film_bounds", bounds(p + ".enlarger_film_bounds", v.enlarger_film_bounds)}, {"film_reference", {{"baseline_raw_rgb", array(p + ".film_reference.baseline_raw_rgb", v.film_reference.baseline_raw_rgb, {3}, "RGB")}, {"compensated_raw_rgb", array(p + ".film_reference.compensated_raw_rgb", v.film_reference.compensated_raw_rgb, {3}, "RGB")}, {"baseline_density_cmy", array(p + ".film_reference.baseline_density_cmy", v.film_reference.baseline_density_cmy, {3}, "CMY")}, {"compensated_density_cmy", array(p + ".film_reference.compensated_density_cmy", v.film_reference.compensated_density_cmy, {3}, "CMY")}, {"meter_scale", scalar(v.film_reference.meter_scale)}, {"hash", scalar(v.film_reference.hash)}}}, {"optics", recipe_optics(p + ".optics", v.optics)}, {"grain", grain(p + ".grain", v.grain)}, {"effects", effects(p + ".effects", v.effects)}, {"exposure_tables", spectral(p + ".exposure_tables", v.exposure_tables)}, {"scanner_tables", spectral(p + ".scanner_tables", v.scanner_tables)}, {"s_inverse", array(p + ".s_inverse", v.s_inverse, {3, 3}, "row-major")}, {"midgray_dwg_rgb", array(p + ".midgray_dwg_rgb", v.midgray_dwg_rgb, {3}, "RGB")}, {"raw_midgray", array(p + ".raw_midgray", v.raw_midgray, {3}, v.raw_midgray_bgr != 0 ? "BGR" : "RGB")}, {"raw_green", scalar(v.raw_green)}, {"midgray_scale", scalar(v.midgray_scale)}, {"raw_midgray_bgr", scalar(v.raw_midgray_bgr)}, {"input_hull", std::move(hull)}, {"output_boundary", std::move(boundary)}, {"scanner_lut", std::move(lut)}, {"scanner_color", scanner_color(p + ".scanner_color", v.scanner_color)}, {"scanner_correction", std::move(corr)}, {"scanner_post_effects", std::move(post)}, {"output_color", std::move(out_color)}, {"print", print_value(p + ".print", v.print)}, {"print_gamma", scalar(v.print_gamma)}, {"print_filter_main_cmy_cc", array(p + ".print_filter_main_cmy_cc", v.print_filter_main_cmy_cc, {3}, "CMY/CC")}, {"print_filter_preflash_cmy_cc", array(p + ".print_filter_preflash_cmy_cc", v.print_filter_preflash_cmy_cc, {3}, "CMY/CC")}, {"print_normalization_mode", scalar(v.print_normalization_mode)}, {"normalize_print_exposure", scalar(v.normalize_print_exposure)}, {"print_exposure_compensation", scalar(v.print_exposure_compensation)}, {"calibration", scalar(v.calibration)}, {"neutral_cmy_cc", array(p + ".neutral_cmy_cc", v.neutral_cmy_cc, {3}, "CMY/CC")}, {"print_source_illuminant", span_node(p + ".print_source_illuminant", v.print_source_illuminant, "wavelength")}, {"print_balance", std::move(bal)}, {"scanner_recipe_hash", scalar(v.scanner_recipe_hash)}, {"print_recipe_hash", scalar(v.print_recipe_hash)}, {"print_filters_hash", scalar(v.print_filters_hash)}, {"print_exposure_hash", scalar(v.print_exposure_hash)}, {"print_illuminant_hash", scalar(v.print_illuminant_hash)}, {"print_curves_hash", scalar(v.print_curves_hash)}};
    }

    Json params(const FjPreparationParams& v) {
        (void)bound_preparation_params(v);
        (void)bound_halation_params(v.scatter_halation);
        const std::string p = "params";
        Json o;
#define S(n) o[#n] = scalar(v.n)
#define A(n, sh, ord) o[#n] = array(p + "." #n, v.n, sh, ord)
        S(print_exposure);
        S(print_preflash_exposure);
        S(print_gamma_factor);
        S(preflash_m_filter_cc);
        S(preflash_y_filter_cc);
        S(print_shadow_compensation_factor);
        S(print_shadow_compensation_density);
        S(print_shadow_compensation_transition);
        S(glare_percent);
        S(glare_roughness);
        S(glare_blur_sigma_px);
        S(camera_exposure_compensation_ev);
        S(camera_film_format_long_edge_mm);
        S(film_gamma_factor);
        S(scanner_lens_blur_sigma_px);
        S(scanner_black_level);
        S(scanner_white_level);
        S(gate_weave_amount);
        A(scanner_unsharp_mask, {2}, "sigma/amount");
        A(print_ui_ymc_cc, {3}, "UI-YMC");
        A(camera_filter_uv, {3}, "UV");
        A(camera_filter_ir, {3}, "IR");
        o["camera_diffusion"] = params_diffusion(p + ".camera_diffusion", v.camera_diffusion);
        o["enlarger_diffusion"] = params_diffusion(p + ".enlarger_diffusion", v.enlarger_diffusion);
        o["dir_couplers"] = params_dir(p + ".dir_couplers", v.dir_couplers);
        o["film_profile_key"] = string_view(v.film_profile_key);
        o["print_profile_key"] = string_view(v.print_profile_key);
#define S(n) o[#n] = scalar(v.n)
        S(spectral_upsampling_mode);
        S(ref_ill);
        S(enl_ill);
        S(normalize_print_exposure);
        S(print_exposure_compensation);
        S(input_color_space);
        S(input_cctf_decoding);
        S(input_compression_enabled);
        S(hanatos2025_adaptation_window);
        S(hanatos2025_adaptation_surface);
        S(camera_auto_exposure_enabled);
        S(camera_metering_method);
        o["scatter_halation"] = {{"active", scalar(v.scatter_halation.active)}, {"scatter_amount", scalar(v.scatter_halation.scatter_amount)}, {"scatter_spatial_scale", scalar(v.scatter_halation.scatter_spatial_scale)}, {"halation_amount", scalar(v.scatter_halation.halation_amount)}, {"halation_spatial_scale", scalar(v.scatter_halation.halation_spatial_scale)}};
        S(scanner_black_correction);
        S(scanner_white_correction);
        S(scanner_lut_resolution);
        S(output_color_space);
        S(output_cctf_encoding);
        S(output_gamut_compression_enabled);
        o["grain"] = params_grain(p + ".grain", v.grain);
        S(film_dust_amount);
        S(film_scratch_amount);
        S(gate_dust_amount);
        S(gate_scratch_amount);
        S(scan_route);
        S(glare_active);
        S(camera_filter_override);
#undef S
#undef A
        return o;
    }

    void compare(const Json& expected, const Json& actual) {
        try {
            check_node(expected, "expected");
            check_node(actual, "actual");
            compare_nodes(expected, actual, "record");
        } catch (const nlohmann::json::exception& error) {
            throw std::runtime_error(std::string("oracle record JSON is malformed: ") + error.what());
        }
    }

} // namespace PreparedStateOracle
