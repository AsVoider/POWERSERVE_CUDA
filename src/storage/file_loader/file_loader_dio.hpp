#pragma once

#include "common/logger.hpp"
#include "fmt/core.h"
#include "storage/file_loader.hpp"

#include <cstddef>
#include <cstdlib>
#include <fcntl.h>
#include <filesystem>
#include <new>
#include <span>
#include <sys/stat.h>
#include <unistd.h>

namespace smart::storage {

class FileLoaderDIO final : public FileLoader {
public:
    static constexpr int BUFFER_ALIGNMENT = 4096;

private:
    std::span<std::byte> m_buffer;

public:
    FileLoaderDIO(const std::filesystem::path &file_path) : FileLoader(file_path) {
        // open file with O_DIRECT for direct I/O
        const int fd = open(m_file_path.c_str(), O_RDONLY | O_DIRECT);
        m_file_handle.reset(fd);
    }

    ~FileLoaderDIO() noexcept override {
        unload();
    }

    FileLoaderDIO(const FileLoaderDIO &other) = delete;

    FileLoaderDIO(FileLoaderDIO &&other) noexcept : FileLoader(std::move(other)), m_buffer(other.m_buffer) {
        other.m_buffer = {};
    }

    FileLoaderDIO &operator=(const FileLoaderDIO &other) = delete;

    FileLoaderDIO &operator=(FileLoaderDIO &&other) noexcept {
        if (this != &other) {
            FileLoader::operator=(std::move(other));

            unload();
            m_buffer = other.m_buffer;
        }
        return *this;
    }

public:
    void load() override {
        if (!m_buffer.empty()) {
            SMART_LOG_WARN("trying to load a buffer twice");
            unload();
        }

        struct stat file_stat;
        {
            const int ret = fstat(m_file_handle.m_fd, &file_stat);
            SMART_ASSERT(ret == 0, "failed to fstat file {}", m_file_path);
        }

        const size_t file_size = file_stat.st_size;

        /*
             * Allocate aligned buffer
             */
        const size_t aligned_file_size = align_ceil(file_size);
        std::byte *buffer_ptr          = new (std::align_val_t{BUFFER_ALIGNMENT}) std::byte[aligned_file_size];
        SMART_ASSERT(buffer_ptr != nullptr, "failed to allocate buffer of size {}", aligned_file_size);
        m_buffer = {buffer_ptr, file_size};

        /*
             * Read the whole file into the buffer
             */
        {
            const ssize_t ret = pread(m_file_handle.m_fd, buffer_ptr, aligned_file_size, 0);
            SMART_ASSERT(
                ret == static_cast<ssize_t>(file_size),
                "faild to read {} bytes from file {} (ret = {})",
                file_size,
                m_file_path,
                ret
            );
        }
    }

    void unload() override {
        if (!m_buffer.empty()) {
            delete[] m_buffer.data();
        }
        m_buffer = {};
    }

    std::span<std::byte> get_buffer(const bool implicit_load = true) override {
        if (implicit_load) {
            load();
        }
        return m_buffer;
    }

    FileLoaderMethod get_method() const override {
        return FileLoaderMethod::BIO;
    }

public:
    template <typename T>
    constexpr static T align_ceil(T val) {
        return (val + BUFFER_ALIGNMENT - 1) / BUFFER_ALIGNMENT * BUFFER_ALIGNMENT;
    }
};

} // namespace smart::storage
