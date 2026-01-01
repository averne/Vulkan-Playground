#pragma once

#include <cstdio>
#include <algorithm>
#include <string_view>
#include <print>
#include <utility>

#include <vulkan/vulkan.hpp>

namespace util {

#define _CAT(x, y) x ## y
#define  CAT(x, y) _CAT(x, y)
#define _STRING(x) #x
#define  STRING(x) _STRING(x)

#define VAR_ANONYMOUS CAT(var, __COUNTER__)

#define SCOPEGUARD(f) auto VAR_ANONYMOUS = ::util::ScopeGuard(f)

#define UNUSED(...) ::util::unused(__VA_ARGS__)

#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((*arr)))

#define VK_CHECK(...) ({                                               \
    if (auto _res_ = (__VA_ARGS__); _res_ !=  vk::Result::eSuccess) {          \
        std::printf(STRING(__VA_ARGS__) ": error %s\n", vk::to_string(_res_).data());  \
        return -1;                                                      \
    }                                                                   \
})

#define VK_CHECK_RV(...) ({                                            \
    auto &&[_res_, _val_] = (__VA_ARGS__);                                     \
    if (_res_ !=  vk::Result::eSuccess) {                               \
        std::printf(STRING(__VA_ARGS__) ": error %s\n", vk::to_string(_res_).data());  \
        return -1;                                                      \
    }                                                                   \
    std::move(_val_);                                                   \
})

template <typename F>
struct ScopeGuard {
    [[nodiscard]] ScopeGuard(F &&f): f(std::move(f)) { }

    ScopeGuard(const ScopeGuard &) = delete;
    ScopeGuard &operator =(const ScopeGuard &) = delete;

    ~ScopeGuard() {
        if (this->want_run)
            this->f();
    }

    void cancel() {
        this->want_run = false;
    }

    private:
        bool want_run = true;
        F f;
};

void unused(auto &&...args) {
    (static_cast<void>(args), ...);
}

constexpr auto align_down(auto v, auto a) {
    return v & ~(a - 1);
}

constexpr auto align_up(auto v, auto a) {
    return align_down(v + a - 1, a);
}

constexpr auto bit(auto bit) {
    return static_cast<decltype(bit)>(1) << bit;
}

constexpr auto mask(auto bit) {
    return (static_cast<decltype(bit)>(1) << bit) - 1;
}

std::uint32_t find_in_family(auto &&family, auto &&pred) {
    auto idx = -1;
    auto it = std::ranges::find_if(family, [&idx, &pred](auto &&v) {
        return ++idx, pred(std::move(v));
    });
    return (it != family.end()) ? idx : -1;
}

template <typename T>
static inline int read_whole_file(T &container, std::string_view path, std::string_view mode = "rb") {
    std::FILE *fp = std::fopen(path.data(), mode.data());
    if (!fp) {
        std::println("Failed to open {}", path);
        return -1;
    }

    SCOPEGUARD([fp] { std::fclose(fp); });

    std::fseek(fp, 0, SEEK_END);
    std::size_t fsize = std::ftell(fp);
    std::rewind(fp);

    auto atom_sz = sizeof(typename T::value_type);
    container.resize(util::align_up(fsize, atom_sz) / atom_sz);

    if (auto read = std::fread(container.data(), 1, container.size() * atom_sz, fp); read != fsize) {
        std::println("Failed to read {}: got {}, expected {}", path, read, fsize);
        return -1;
    }

    return 0;
}

} // namespace util
