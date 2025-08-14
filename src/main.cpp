#include <cstdint>
#include <numeric>
#include <numbers>
#include <random>
#include <ranges>

#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.hpp>

#include "util.hpp"

const char *layers[] = {
#ifdef DEBUG
    "VK_LAYER_KHRONOS_validation",
#endif
};

template <typename T>
using MacroBlock       = std::array<T, 8*8>;
using MacroBlockFloat  = MacroBlock<float>;
using MacroBlockDouble = MacroBlock<double>;

// Naive iDCT, see Apple ProRes Bitstream Syntax and Decoding Process, section 7.4 "Inverse Transform"
void idct(MacroBlockDouble &in, MacroBlockDouble &out) {
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            double sum = 0.0;
            for (int u = 0; u < 8; ++u) {
                for (int v = 0; v < 8; ++v) {
                    constexpr auto inv_sqrt2 = std::numbers::sqrt2 / 2.0;
                    sum += (u ? 1 : inv_sqrt2) * (v ? 1 : inv_sqrt2) * in[v * 8 + u] *
                           std::cos((2.0 * x + 1.0) * u * std::numbers::pi / 16.0) *
                           std::cos((2.0 * y + 1.0) * v * std::numbers::pi / 16.0);
                }
                out[y * 8 + x] = sum / 4.0;
            }
        }
    }
}

int main(int argc, char **argv) {
    // Create device and initialize base objects
    auto app_info             = vk::ApplicationInfo("ProRes-iDCT", 0, "", 0, VK_API_VERSION_1_2);
    auto instance_create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info, ARRAY_SIZE(layers), layers);

    auto instance = VK_CHECK_RV(vk::createInstanceUnique(instance_create_info));
    auto physdev  = VK_CHECK_RV(instance->enumeratePhysicalDevices()).front();

    auto dev_props = physdev.getProperties();
    std::println("Starting on device {}, api {}.{}.{}, driver {}.{}.{}", dev_props.deviceName.data(),
                 VK_VERSION_MAJOR(dev_props.apiVersion),    VK_VERSION_MINOR(dev_props.apiVersion),    VK_VERSION_PATCH(dev_props.apiVersion),
                 VK_VERSION_MAJOR(dev_props.driverVersion), VK_VERSION_MINOR(dev_props.driverVersion), VK_VERSION_PATCH(dev_props.driverVersion));

    auto queue_idx = util::find_in_family(physdev.getQueueFamilyProperties(), [](auto &&e) {
        return !!(e.queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer));
    });

    auto mem_props = physdev.getMemoryProperties();
    auto host_buf_idx = util::find_in_family(mem_props.memoryTypes, [](auto &&e) {
        return !!(e.propertyFlags & vk::MemoryPropertyFlagBits::eHostCoherent);
    });

    auto dev_buf_idx = util::find_in_family(mem_props.memoryTypes, [](auto &&e) {
        return !!(e.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal);
    });

    assert(queue_idx >= 0 && host_buf_idx >= 0 && dev_buf_idx >= 0);

    std::array queue_prios = { 1.0f };
    auto queue_create_info = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queue_idx, queue_prios);
    auto dev_create_info = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queue_create_info);

    auto dev   = VK_CHECK_RV(physdev.createDeviceUnique(dev_create_info));
    auto queue = dev->getQueue(queue_idx, 0);

    auto command_pool = VK_CHECK_RV(dev->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                                                           queue_idx)));
    auto cmdbuf       = VK_CHECK_RV(dev->allocateCommandBuffers(vk::CommandBufferAllocateInfo(*command_pool, vk::CommandBufferLevel::ePrimary, 1))).front();

    auto fence = VK_CHECK_RV(dev->createFenceUnique(vk::FenceCreateInfo()));

    // Create memory buffers
    constexpr auto buf_size = sizeof(MacroBlockFloat);

    auto inbuf  = VK_CHECK_RV(dev->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), buf_size,
                                                                           vk::BufferUsageFlagBits::eStorageBuffer |
                                                                           vk::BufferUsageFlagBits::eTransferDst)));
    auto outbuf = VK_CHECK_RV(dev->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), buf_size,
                                                                           vk::BufferUsageFlagBits::eStorageBuffer |
                                                                           vk::BufferUsageFlagBits::eTransferSrc   |
                                                                           vk::BufferUsageFlagBits::eTransferDst)));

    auto inmem  = VK_CHECK_RV(dev->allocateMemoryUnique(vk::MemoryAllocateInfo(dev->getBufferMemoryRequirements(*inbuf ).size,
                                                                               host_buf_idx)));
    auto outmem = VK_CHECK_RV(dev->allocateMemoryUnique(vk::MemoryAllocateInfo(dev->getBufferMemoryRequirements(*outbuf).size,
                                                                               dev_buf_idx )));

    VK_CHECK(dev->bindBufferMemory(*inbuf,  *inmem,  0));
    VK_CHECK(dev->bindBufferMemory(*outbuf, *outmem, 0));

    auto *addr = VK_CHECK_RV(dev->mapMemory(*inmem, 0, buf_size));
    SCOPEGUARD([&] { dev->unmapMemory(*inmem); });

    // Randomly initialize the reference data and the shader input data
    // We use a uniform quarter-integer distribution in the range [-2048.0, 2048) as described in
    // Apple ProRes Bitstream Syntax and Decoding Process, annex A
    auto rand = std::mt19937(std::random_device()());
    auto dist = std::uniform_int_distribution<int>(-2048*4, 2048*4-1);

    MacroBlockDouble data = {};
    std::ranges::generate(data, [&] { return dist(rand) / 4.0; });

    auto vals = std::span(static_cast<float *>(addr), 64);
    std::ranges::transform(data, vals.begin(), [](auto in) { return in; });

    // Load shader code
    std::vector<std::uint32_t> shader_bin;
    util::read_whole_file(shader_bin, "build/prores-idct.spv");
    if (shader_bin.empty())
        return -1;

    auto shader = VK_CHECK_RV(dev->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
                                                                                       shader_bin.size() * sizeof(decltype(shader_bin)::value_type),
                                                                                       shader_bin.data())));

    // Set up descriptors
    auto desc_sizes = vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, 2);
    auto desc_pool = VK_CHECK_RV(dev->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                                                                              1, desc_sizes)));

    std::array desc_bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute)
    };
    auto desc_layout = VK_CHECK_RV(dev->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo(vk::DescriptorSetLayoutCreateFlags(), desc_bindings)));
    auto desc_sets = VK_CHECK_RV(dev->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(*desc_pool, *desc_layout)));
    auto descriptors = std::move(desc_sets.front());

	auto in_info = vk::DescriptorBufferInfo(*inbuf, 0, buf_size), out_info = vk::DescriptorBufferInfo(*outbuf, 0, buf_size);
    dev->updateDescriptorSets({
        {*descriptors, 0, 0, vk::DescriptorType::eStorageBuffer, nullptr, in_info },
        {*descriptors, 1, 0, vk::DescriptorType::eStorageBuffer, nullptr, out_info},
    }, nullptr);

    // Set up pipeline
    auto pipeline_layout = VK_CHECK_RV(dev->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), *desc_layout)));

    auto pipeline_create_info = vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                                              vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                                                vk::ShaderStageFlagBits::eCompute, *shader, "main"),
                                                              *pipeline_layout);
    auto pipeline = VK_CHECK_RV(dev->createComputePipelineUnique(nullptr, pipeline_create_info));

    auto query_pool = VK_CHECK_RV(dev->createQueryPoolUnique(vk::QueryPoolCreateInfo(vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 2)));

    // Finally, write commands and kickoff
    vk::BufferMemoryBarrier transfer_barrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                             VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *outbuf, 0, buf_size);

    VK_CHECK(cmdbuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
    cmdbuf.resetQueryPool(*query_pool, 0, 2);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmdbuf.bindDescriptorSets(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, *descriptors, nullptr);
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *query_pool, 0);
    cmdbuf.dispatch(1, 1, 1);
    cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *query_pool, 1);
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion, nullptr, transfer_barrier, nullptr);
    cmdbuf.copyBuffer(*outbuf, *inbuf, vk::BufferCopy(0, 0, buf_size));
    VK_CHECK(cmdbuf.end());

    VK_CHECK(queue.submit(vk::SubmitInfo(nullptr, nullptr, cmdbuf), *fence));
    VK_CHECK(dev->waitForFences(*fence, VK_TRUE, UINT64_MAX));

    auto res = VK_CHECK_RV(dev->getQueryPoolResults<std::uint64_t>(*query_pool, 0, 2, 2 * sizeof(std::uint64_t),
                                                                   sizeof(std::uint64_t), vk::QueryResultFlagBits::e64));
    std::println("Kernel execution time: {}us", (res[1] - res[0]) * dev_props.limits.timestampPeriod / 1e3);

    MacroBlockDouble soft;
    idct(data, soft);

    std::println("Reference reconstruction:");
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[ 0], soft[ 1], soft[ 2], soft[ 3], soft[ 4], soft[ 5], soft[ 6], soft[ 7]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[ 8], soft[ 9], soft[10], soft[11], soft[12], soft[13], soft[14], soft[15]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[16], soft[17], soft[18], soft[19], soft[20], soft[21], soft[22], soft[23]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[24], soft[25], soft[26], soft[27], soft[28], soft[29], soft[30], soft[31]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[32], soft[33], soft[34], soft[35], soft[36], soft[37], soft[38], soft[39]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[40], soft[41], soft[42], soft[43], soft[44], soft[45], soft[46], soft[47]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[48], soft[49], soft[50], soft[51], soft[52], soft[53], soft[54], soft[55]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        soft[56], soft[57], soft[58], soft[59], soft[60], soft[61], soft[62], soft[63]);

    std::println("Accelerated reconstruction:");
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[ 0], vals[ 1], vals[ 2], vals[ 3], vals[ 4], vals[ 5], vals[ 6], vals[ 7]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[ 8], vals[ 9], vals[10], vals[11], vals[12], vals[13], vals[14], vals[15]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[16], vals[17], vals[18], vals[19], vals[20], vals[21], vals[22], vals[23]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[24], vals[25], vals[26], vals[27], vals[28], vals[29], vals[30], vals[31]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[32], vals[33], vals[34], vals[35], vals[36], vals[37], vals[38], vals[39]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[40], vals[41], vals[42], vals[43], vals[44], vals[45], vals[46], vals[47]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[48], vals[49], vals[50], vals[51], vals[52], vals[53], vals[54], vals[55]);
    std::println("{:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f} {:+10.4f}",
        vals[56], vals[57], vals[58], vals[59], vals[60], vals[61], vals[62], vals[63]);

    // Validate iDCT calculation, see Apple ProRes Bitstream Syntax and Decoding Process,
    // annex A "IDCT Implementation Accuracy Qualification"
    MacroBlockDouble delta = {}, me = {}, mse = {};
    std::ranges::transform(std::views::zip(soft, vals), delta.begin(),
                           [](auto in) { return std::get<0>(in) - std::get<1>(in); });

    std::ranges::transform(delta, me .begin(), [&](auto in) { return in      / delta.size(); });
    std::ranges::transform(delta, mse.begin(), [&](auto in) { return in * in / delta.size(); });

    auto ppe  = std::ranges::max_element(delta);
    auto pme  = std::ranges::max_element(me);
    auto pmse = std::ranges::max_element(mse);

    auto ome  = std::ranges::fold_left(me,  0, std::plus()) / delta.size();
    auto omse = std::ranges::fold_left(mse, 0, std::plus()) / delta.size();

    std::println("Residual error: peak delta {:.3e}, peak mean {:.3e}, peak squared {:.3e}, "
                 "overall mean {:.3e}, overall squared {:.3e}",
                 *ppe, *pme, *pmse, ome, omse);

    if (*ppe >= 0.15 || *pme >= 0.0015 || *pmse >= 0.002 || ome >= 0.00015 || omse >= 0.002)
        std::println("Residual error exceeds acceptable threshold!");

    return 0;
}
