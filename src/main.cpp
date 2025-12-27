#include <cstdint>
#include <chrono>

#include <sys/stat.h>

#define VULKAN_HPP_NO_NODISCARD_WARNINGS
#define VULKAN_HPP_NO_EXCEPTIONS
#include <vulkan/vulkan.hpp>

#include "util.hpp"

const char *layers[] = {
#ifdef DEBUG
    "VK_LAYER_KHRONOS_validation",
#endif
};

struct PushData {
    vk::DeviceAddress in_addr;
    vk::DeviceAddress out_addr;
    vk::DeviceAddress tmp_addr;
    uint32_t          array_size;
};

int main(int argc, char **argv) {
    // Create device and initialize base objects
    auto app_info             = vk::ApplicationInfo("vk-test", 0, "", 0, VK_API_VERSION_1_2);
    auto instance_create_info = vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &app_info, ARRAY_SIZE(layers), layers);

    auto instance = VK_CHECK_RV(vk::createInstanceUnique(instance_create_info));
    auto physdev  = VK_CHECK_RV(instance->enumeratePhysicalDevices()).front();

    auto queue_props = physdev.getQueueFamilyProperties();
    auto queue_idx = util::find_in_family(queue_props, [](auto &&e) {
        return !!(e.queueFlags & (vk::QueueFlagBits::eCompute | vk::QueueFlagBits::eTransfer));
    });

    auto mem_props = physdev.getMemoryProperties();
    auto host_buf_idx = util::find_in_family(mem_props.memoryTypes, [](auto &&e) {
        return !!(e.propertyFlags & (vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached));
    });

    auto dev_buf_idx = util::find_in_family(mem_props.memoryTypes, [](auto &&e) {
        return !!(e.propertyFlags & (vk::MemoryPropertyFlagBits::eDeviceLocal));
    });

    assert(queue_idx >= 0 && host_buf_idx >= 0 && dev_buf_idx >= 0);

    auto feats = physdev.getFeatures2();
    feats.features
        .setShaderInt16(vk::True)
        .setShaderInt64(vk::True);

    auto subgroup_props = vk::PhysicalDeviceSubgroupProperties();
    auto props = vk::PhysicalDeviceProperties2({}, &subgroup_props);
    physdev.getProperties2(&props);

    auto feats12 = vk::PhysicalDeviceVulkan12Features()
        .setBufferDeviceAddress(vk::True)
        .setShaderInt8(vk::True)
        .setStorageBuffer8BitAccess(vk::True);
    feats.setPNext(&feats12);

    std::println("Starting on device {}, api {}.{}.{}, driver {}.{}.{}", props.properties.deviceName.data(),
                 VK_VERSION_MAJOR(props.properties.apiVersion),    VK_VERSION_MINOR(props.properties.apiVersion),    VK_VERSION_PATCH(props.properties.apiVersion),
                 VK_VERSION_MAJOR(props.properties.driverVersion), VK_VERSION_MINOR(props.properties.driverVersion), VK_VERSION_PATCH(props.properties.driverVersion));
    std::println("Subgroup support: size {}, ops mask {:#x}", subgroup_props.subgroupSize, static_cast<std::uint32_t>(subgroup_props.supportedOperations));

    std::array queue_prios = { 1.0f };
    auto queue_create_info = vk::DeviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), queue_idx, queue_prios);
    auto dev_create_info = vk::DeviceCreateInfo(vk::DeviceCreateFlags(), queue_create_info, nullptr, nullptr, nullptr, &feats);

    auto dev   = VK_CHECK_RV(physdev.createDeviceUnique(dev_create_info));
    auto queue = dev->getQueue(queue_idx, 0);

    auto command_pool = VK_CHECK_RV(dev->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                                                                           queue_idx)));
    auto cmdbuf       = VK_CHECK_RV(dev->allocateCommandBuffers(vk::CommandBufferAllocateInfo(*command_pool, vk::CommandBufferLevel::ePrimary, 1))).front();

    auto fence = VK_CHECK_RV(dev->createFenceUnique(vk::FenceCreateInfo()));

    auto *fp = std::fopen("scripts/in.bin", "rb");
    SCOPEGUARD([&fp] { std::fclose(fp); });
    if (!fp) {
        std::println("Failed to open input: {}", std::strerror(errno));
        return 1;
    }

    struct stat st;
    if (auto rc = ::fstat(::fileno(fp), &st); rc < 0) {
        std::println("Failed to tell file size: {}", std::strerror(errno));
        return 1;
    }

    auto array_size = st.st_size;
    auto inbuf_size = array_size * sizeof(std::uint32_t),
        outbuf_size = array_size * sizeof(std::uint32_t),
        tmpbuf_size = array_size * sizeof(std::uint64_t);

    auto inbuf  = VK_CHECK_RV(dev->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), inbuf_size,
                                                                           vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                                                           vk::BufferUsageFlagBits::eStorageBuffer       |
                                                                           vk::BufferUsageFlagBits::eTransferDst)));
    auto outbuf = VK_CHECK_RV(dev->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), outbuf_size,
                                                                           vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                                                           vk::BufferUsageFlagBits::eStorageBuffer       |
                                                                           vk::BufferUsageFlagBits::eTransferSrc)));
    auto tmpbuf = VK_CHECK_RV(dev->createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), tmpbuf_size,
                                                                           vk::BufferUsageFlagBits::eShaderDeviceAddress |
                                                                           vk::BufferUsageFlagBits::eStorageBuffer       |
                                                                           vk::BufferUsageFlagBits::eTransferDst)));

    auto memalloc_pnext = vk::MemoryAllocateFlagsInfoKHR(vk::MemoryAllocateFlagBits::eDeviceAddress);
    auto inmem  = VK_CHECK_RV(dev->allocateMemoryUnique(vk::MemoryAllocateInfo(dev->getBufferMemoryRequirements(*inbuf ).size,
                                                                               host_buf_idx, &memalloc_pnext)));
    auto outmem = VK_CHECK_RV(dev->allocateMemoryUnique(vk::MemoryAllocateInfo(dev->getBufferMemoryRequirements(*outbuf).size,
                                                                               dev_buf_idx,  &memalloc_pnext)));
    auto tmpmem = VK_CHECK_RV(dev->allocateMemoryUnique(vk::MemoryAllocateInfo(dev->getBufferMemoryRequirements(*tmpbuf).size,
                                                                               dev_buf_idx,  &memalloc_pnext)));

    VK_CHECK(dev->bindBufferMemory(*inbuf,  *inmem,  0));
    VK_CHECK(dev->bindBufferMemory(*outbuf, *outmem, 0));
    VK_CHECK(dev->bindBufferMemory(*tmpbuf, *tmpmem, 0));

    auto inshdaddr  = dev->getBufferAddress(vk::BufferDeviceAddressInfo(*inbuf));
    auto outshdaddr = dev->getBufferAddress(vk::BufferDeviceAddressInfo(*outbuf));
    auto tmpshdaddr = dev->getBufferAddress(vk::BufferDeviceAddressInfo(*tmpbuf));

    auto *addr = VK_CHECK_RV(dev->mapMemory(*inmem, 0, inbuf_size));
    SCOPEGUARD([&] { dev->unmapMemory(*inmem); });

    if (auto rd = std::fread(addr, sizeof(std::uint8_t), array_size, fp); rd != array_size) {
        std::println("Failed to read input file: {} (got {}, expected {})", std::strerror(errno), rd, array_size);
        return 1;
    }

    VK_CHECK(dev->flushMappedMemoryRanges(vk::MappedMemoryRange(*inmem, 0, inbuf_size)));

    // Load shader code
    std::vector<std::uint32_t> shader_bin;
    util::read_whole_file(shader_bin, "build/prefix_scan.spv");
    if (shader_bin.empty())
        return -1;

    auto shader = VK_CHECK_RV(dev->createShaderModuleUnique(vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(),
                                                                                       shader_bin.size() * sizeof(decltype(shader_bin)::value_type),
                                                                                       shader_bin.data())));
    // Set up pipeline
    auto push_lyt = vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0,  sizeof(PushData));
    auto pipeline_layout = VK_CHECK_RV(dev->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo(vk::PipelineLayoutCreateFlags(), nullptr, push_lyt)));

    auto pipeline_create_info = vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(),
                                                              vk::PipelineShaderStageCreateInfo(vk::PipelineShaderStageCreateFlags(),
                                                                                                vk::ShaderStageFlagBits::eCompute, *shader, "main"),
                                                              *pipeline_layout);
    auto pipeline = VK_CHECK_RV(dev->createComputePipelineUnique(nullptr, pipeline_create_info));

    auto has_timestamps = queue_props[queue_idx].timestampValidBits > 0;
    auto query_pool = VK_CHECK_RV(dev->createQueryPoolUnique(vk::QueryPoolCreateInfo(vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, 2)));

    // Finally, write commands and kickoff
    auto status_barrier = vk::BufferMemoryBarrier(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eShaderWrite,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *tmpbuf, 0, tmpbuf_size),
         result_barrier = vk::BufferMemoryBarrier(vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite,
                                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, *outbuf, 0, outbuf_size);

    VK_CHECK(cmdbuf.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)));
    cmdbuf.fillBuffer(*tmpbuf, 0, tmpbuf_size, 0);
    cmdbuf.resetQueryPool(*query_pool, 0, 2);
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
    cmdbuf.pushConstants<PushData>(*pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
        PushData(inshdaddr, outshdaddr, tmpshdaddr, array_size));
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader,
                           vk::DependencyFlagBits::eByRegion, nullptr, status_barrier, nullptr);
    if (has_timestamps) cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, *query_pool, 0);
    cmdbuf.dispatch(array_size >> 8, 1, 1);
    if (has_timestamps) cmdbuf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, *query_pool, 1);
    cmdbuf.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer,
                           vk::DependencyFlagBits::eByRegion, nullptr, result_barrier, nullptr);
    cmdbuf.copyBuffer(*outbuf, *inbuf, vk::BufferCopy(0, 0, outbuf_size));
    VK_CHECK(cmdbuf.end());

    auto start = std::chrono::system_clock::now();
    VK_CHECK(queue.submit(vk::SubmitInfo(nullptr, nullptr, cmdbuf), *fence));
    VK_CHECK(dev->waitForFences(*fence, VK_TRUE, UINT64_MAX));
    auto end = std::chrono::system_clock::now();

    std::uint64_t time_us;
    if (has_timestamps) {
        auto res = VK_CHECK_RV(dev->getQueryPoolResults<std::uint64_t>(*query_pool, 0, 2, 2 * sizeof(std::uint64_t),
                                                                       sizeof(std::uint64_t), vk::QueryResultFlagBits::e64));
        time_us = (res[1] - res[0]) * props.properties.limits.timestampPeriod / 1e3;
    } else {
        time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    std::println("Kernel execution time: {}us", time_us);

    VK_CHECK(dev->invalidateMappedMemoryRanges(vk::MappedMemoryRange(*inmem, 0, outbuf_size)));

    auto *fpout = std::fopen("scripts/out.bin", "wb");
    SCOPEGUARD([&fpout] { std::fclose(fpout); });
    if (auto wr = std::fwrite(addr, sizeof(uint8_t), outbuf_size, fpout); wr != outbuf_size) {
        std::println("Failed to write output file: {} (got {}, expected {})", std::strerror(errno), wr, outbuf_size);
        return 1;
    }

    return 0;
}
