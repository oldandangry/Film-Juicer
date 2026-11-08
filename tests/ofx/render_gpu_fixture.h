#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "instance_host.h"
#include "../ffi/print_illuminant_fixture.h"
#include "juicer_ofx_api.h"
#include "ofxImageEffectExt.h"

namespace ofx_test {
    inline void require_render(bool condition, const char* message) {
        if (!condition)
            throw std::runtime_error(message);
    }

    inline void require_cuda(cudaError_t status, const char* operation) {
        if (status != cudaSuccess)
            throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }

    struct ImageLayout {
        int width;
        int height;
        int components;
        int source_padding = 5;
        int destination_padding = 5;
        int border = 0;
    };

    // The reference input is authored independently in ProcessorReference.
    class DeviceImages {
    public:
        explicit DeviceImages(ImageLayout layout)
            : layout_(checked(layout)), pitch_(layout_.width * layout_.components + layout_.source_padding),
              destination_pitch_((layout_.width + 2 * layout_.border) * layout_.components + layout_.destination_padding),
              source_host_(static_cast<std::size_t>(pitch_) * static_cast<std::size_t>(layout_.height), -7.0f),
              destination_host_(static_cast<std::size_t>(destination_pitch_) * static_cast<std::size_t>(layout_.height + 2 * layout_.border), kCanary) {
            for (int y = 0; y < layout_.height; ++y)
                for (int x = 0; x < layout_.width; ++x)
                    for (int c = 0; c < layout_.components; ++c)
                        source_host_[static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch_) + static_cast<std::size_t>(x) * static_cast<std::size_t>(layout_.components) + static_cast<std::size_t>(c)] =
                            c == 3 ? 0.125f + static_cast<float>((x + y) % 4) * 0.25f
                                   : static_cast<float>((x + 3 * y + 7 * c) % 19) / 19.0f;
            try {
                require_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "create stream");
                require_cuda(cudaMalloc(&source_, bytes()), "allocate source");
                require_cuda(cudaMalloc(&destination_, destination_bytes()), "allocate destination");
                reset();
            } catch (...) {
                release();
                throw;
            }
        }
        DeviceImages(const DeviceImages&) = delete;
        DeviceImages& operator=(const DeviceImages&) = delete;
        ~DeviceImages() {
            release();
        }

        void reset() {
            std::fill(destination_host_.begin(), destination_host_.end(), kCanary);
            require_cuda(cudaMemcpyAsync(source_, source_host_.data(), bytes(), cudaMemcpyHostToDevice, stream_), "upload source");
            require_cuda(cudaMemcpyAsync(destination_, destination_host_.data(), destination_bytes(), cudaMemcpyHostToDevice, stream_), "initialize destination");
            require_cuda(cudaStreamSynchronize(stream_), "initialize images");
        }
        std::vector<float> read() {
            require_cuda(cudaDeviceSynchronize(), "complete render");
            require_cuda(cudaMemcpy(destination_host_.data(), destination_, destination_bytes(), cudaMemcpyDeviceToHost), "read destination");
            return destination_host_;
        }
        std::vector<float> pixels(const std::vector<float>& pitched) const {
            std::vector<float> result;
            result.reserve(static_cast<std::size_t>(layout_.width) * static_cast<std::size_t>(layout_.height) * static_cast<std::size_t>(layout_.components));
            for (int y = 0; y < layout_.height + 2 * layout_.border; ++y) {
                for (int sample = 0; sample < destination_pitch_; ++sample) {
                    const bool rendered = y >= layout_.border && y < layout_.height + layout_.border &&
                                          sample >= layout_.border * layout_.components && sample < (layout_.border + layout_.width) * layout_.components;
                    const auto index = static_cast<std::size_t>(y) * static_cast<std::size_t>(destination_pitch_) + static_cast<std::size_t>(sample);
                    if (!rendered)
                        require_render(std::bit_cast<std::uint32_t>(pitched[index]) == std::bit_cast<std::uint32_t>(kCanary), "destination outside window changed");
                }
                if (y >= layout_.border && y < layout_.height + layout_.border) {
                    const auto start = pitched.begin() + static_cast<std::ptrdiff_t>(y) * static_cast<std::ptrdiff_t>(destination_pitch_) + static_cast<std::ptrdiff_t>(layout_.border) * static_cast<std::ptrdiff_t>(layout_.components);
                    result.insert(result.end(), start, start + static_cast<std::ptrdiff_t>(layout_.width) * static_cast<std::ptrdiff_t>(layout_.components));
                }
            }
            if (layout_.components == 4)
                for (int y = 0; y < layout_.height; ++y)
                    for (int x = 0; x < layout_.width; ++x) {
                        const auto source_index = static_cast<std::size_t>(y) * static_cast<std::size_t>(pitch_) + static_cast<std::size_t>(x) * static_cast<std::size_t>(layout_.components) + 3;
                        const auto destination_index = static_cast<std::size_t>(y + layout_.border) * static_cast<std::size_t>(destination_pitch_) + static_cast<std::size_t>(x + layout_.border) * static_cast<std::size_t>(layout_.components) + 3;
                        require_render(std::bit_cast<std::uint32_t>(pitched[destination_index]) == std::bit_cast<std::uint32_t>(source_host_[source_index]), "alpha bits changed");
                    }
            return result;
        }
        float* source() const noexcept {
            return source_;
        }
        float* destination() const noexcept {
            return destination_;
        }
        cudaStream_t stream() const noexcept {
            return stream_;
        }
        int row_bytes() const noexcept {
            return pitch_ * static_cast<int>(sizeof(float));
        }
        int destination_row_bytes() const noexcept {
            return destination_pitch_ * static_cast<int>(sizeof(float));
        }
        const ImageLayout& layout() const noexcept {
            return layout_;
        }
        int border() const noexcept {
            return layout_.border;
        }
        std::size_t bytes() const noexcept {
            return source_host_.size() * sizeof(float);
        }
        std::size_t destination_bytes() const noexcept {
            return destination_host_.size() * sizeof(float);
        }
        static constexpr float kCanary = -23.0f;

    private:
        static ImageLayout checked(ImageLayout value) {
            require_render(value.width > 0 && value.width <= 16384 && value.height > 0 && value.height <= 16384 &&
                               (value.components == 3 || value.components == 4) && value.source_padding >= 0 && value.source_padding <= 4096 &&
                               value.destination_padding >= 0 && value.destination_padding <= 4096 && value.border >= 0 && value.border <= 16,
                           "invalid fixture image layout");
            return value;
        }
        void release() noexcept {
            if (stream_)
                (void)cudaStreamSynchronize(stream_);
            if (destination_)
                (void)cudaFree(destination_);
            if (source_)
                (void)cudaFree(source_);
            if (stream_)
                (void)cudaStreamDestroy(stream_);
        }
        ImageLayout layout_;
        int pitch_;
        int destination_pitch_;
        std::vector<float> source_host_;
        std::vector<float> destination_host_;
        float* source_ = nullptr;
        float* destination_ = nullptr;
        cudaStream_t stream_ = nullptr;
    };

    inline std::vector<float> reference_pixels(const std::string& path, std::string_view selected) {
        std::ifstream input(path);
        std::string name;
        std::size_t count = 0;
        while (input >> name >> count) {
            std::vector<float> row(count);
            for (auto& value : row)
                input >> value;
            require_render(static_cast<bool>(input), "incomplete reference pixel row");
            if (name == selected) {
                nlohmann::json words = nlohmann::json::array();
                for (float value : row)
                    words.push_back(std::bit_cast<std::uint32_t>(value));
                auto selected_path = path;
                selected_path += '.';
                selected_path += name;
                const auto selected_words = PrintIlluminantFixtures::apply(selected_path, words);
                require_render(selected_words.size() == row.size(), "reference pixel supplement shape");
                for (std::size_t i = 0; i < row.size(); ++i)
                    row[i] = std::bit_cast<float>(selected_words.at(i).get<std::uint32_t>());
                return row;
            }
        }
        throw std::runtime_error("reference pixel row missing: " + std::string(selected));
    }
    class RenderHost {
    public:
        explicit RenderHost(OfxPlugin* plugin = fj_ofx_plugin(0)) : plugin_(plugin) {
            require_render(plugin_ != nullptr, "candidate descriptor missing");
            plugin_->setHost(host_.binding());
            require_render(call(kOfxActionLoad) == kOfxStatOK, "load candidate");
            loaded_ = true;
            require_render(call(kOfxImageEffectActionDescribeInContext, host_.descriptor_handle(), host_.context_args()) == kOfxStatOK,
                           "describe context");
            effect_ = host_.create_effect();
            require_render(call(kOfxActionCreateInstance, effect_) == kOfxStatOK, "create instance");
            created_ = true;
        }
        RenderHost(const RenderHost&) = delete;
        RenderHost& operator=(const RenderHost&) = delete;
        ~RenderHost() {
            if (created_)
                (void)call(kOfxActionDestroyInstance, effect_);
            if (loaded_)
                (void)call(kOfxActionUnload);
        }
        OfxStatus call(const char* action, const void* handle = nullptr, OfxPropertySetHandle input = nullptr) const {
            return plugin_->mainEntry(action, handle, input, nullptr);
        }
        void configure(DeviceImages& images, OfxPointI origin = {0, 0}) {
            configure_effect(effect_, images, origin);
        }
        void configure_effect(OfxImageEffectHandle effect, DeviceImages& images, OfxPointI origin = {0, 0}) {
            require_render(origin.x >= -10000 && origin.x <= 10000 && origin.y >= -10000 && origin.y <= 10000, "invalid fixture origin");
            const auto& layout = images.layout();
            const int width = layout.width;
            const int height = layout.height;
            const int components = layout.components;
            const int x = origin.x;
            const int y = origin.y;
            host_.set_effect_property(effect, kOfxImageEffectPropFrameRate, {24.0});
            auto* args = host_.action_args();
            host_.set_property(args, kOfxPropTime, {37.0});
            host_.set_property(args, kOfxImageEffectPropRenderScale, {1.0, 1.0});
            host_.set_property(args, kOfxImageEffectPropRenderWindow, {x, y, x + width, y + height});
            host_.set_property(args, kOfxImageEffectPropCudaEnabled, {1});
            host_.set_property(args, kOfxImageEffectPropCudaStream, {static_cast<void*>(images.stream())});
            host_.set_property(args, kOfxImageEffectPropFieldToRender, {std::string(kOfxImageFieldNone)});
            for (const auto* clip : {"Source", "Output"}) {
                const auto set = [&](const char* key, std::vector<InstanceValue> values) {
                    host_.set_image_property(effect, clip, key, std::move(values));
                };
                const bool source = std::string_view(clip) == "Source";
                const int border = source ? 0 : images.border();
                set(kOfxImagePropRowBytes, {source ? images.row_bytes() : images.destination_row_bytes()});
                set(kOfxImagePropPixelAspectRatio, {1.0});
                set(kOfxImageEffectPropComponents, {std::string(components == 3 ? kOfxImageComponentRGB : kOfxImageComponentRGBA)});
                set(kOfxImageEffectPropPixelDepth, {std::string(kOfxBitDepthFloat)});
                set(kOfxImageEffectPropPreMultiplication, {std::string(kOfxImageUnPreMultiplied)});
                set(kOfxImagePropBounds, {x - border, y - border, x + width + border, y + height + border});
                set(kOfxImagePropRegionOfDefinition, {x, y, x + width, y + height});
                set(kOfxImagePropField, {std::string(kOfxImageFieldNone)});
                set(kOfxImagePropUniqueIdentifier, {std::string(clip)});
                set(kOfxImageEffectPropRenderScale, {1.0, 1.0});
                set(kOfxImagePropData, {static_cast<void*>(std::string_view(clip) == "Source" ? images.source() : images.destination())});
                host_.set_clip_region(effect, clip, OfxRectD{static_cast<double>(x), static_cast<double>(y), static_cast<double>(x + width), static_cast<double>(y + height)});
            }
            host_.set_value(effect, "CameraAutoExposure", {0});
            host_.set_value(effect, "ScannerUnsharpMask", {0.0, 0.0});
            host_.set_value(effect, "GateWeaveAmount", {0.0});
            host_.set_value(effect, "HalationActive", {0});
            host_.set_value(effect, "GlareActive", {0});
            host_.set_value(effect, "ScanRoute", {std::string("NegativeDirectScan")});
        }
        void close() {
            if (created_) {
                require_render(call(kOfxActionDestroyInstance, effect_) == kOfxStatOK, "destroy instance");
                created_ = false;
            }
            if (loaded_) {
                require_render(call(kOfxActionUnload) == kOfxStatOK, "unload candidate");
                loaded_ = false;
            }
        }
        OfxStatus render() {
            return call(kOfxImageEffectActionRender, effect_, host_.action_args());
        }
        InstanceHost& host() noexcept {
            return host_;
        }
        OfxImageEffectHandle effect() const noexcept {
            return effect_;
        }

    private:
        InstanceHost host_;
        OfxPlugin* plugin_;
        OfxImageEffectHandle effect_ = nullptr;
        bool loaded_ = false;
        bool created_ = false;
    };
} // namespace ofx_test
