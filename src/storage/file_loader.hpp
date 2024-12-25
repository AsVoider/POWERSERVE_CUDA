#pragma once

#include "file.hpp"

#include <cstddef>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <memory>
#include <span>

namespace smart::storage {

enum class FileLoaderMethod : int {
    /// Load file using mmap
    MMap = 0,
    /// Load file using buffered I/O interface
    BIO = 1,
    /// Load file using direct I/O interface
    DIO = 2,
    /// Load file using libuv I/O interface
    UV = 3
};

///
/// Simple loader for file-granularity Reading
/// @note Only reading is supported. Writing file buffer leads to undefined behaviours
///
class FileLoader {
protected:
    /// The path to the file
    std::filesystem::path m_file_path;
    /// The file descriptor
    FileHandle m_file_handle;

public:
    FileLoader(const std::filesystem::path &file_path) : m_file_path(file_path) {}

    virtual ~FileLoader() noexcept = default;

    FileLoader(const FileLoader &other) = delete;

    FileLoader(FileLoader &&other) noexcept = default;

    FileLoader &operator=(const FileLoader &other) = delete;

    FileLoader &operator=(FileLoader &&other) noexcept = default;

public: /* Buffer Operation */
    /*!
         * @brief Load the whole file into buffer
         * @note This may incurs large memory allocation
         */
    virtual void load() = 0;

    /*!
         * @brief Release file buffer while keeping the file handle
         */
    virtual void unload() = 0;

public: /* Getter */
    /*!
         * @brief Get the file buffer 
         * @param[in] implicit_load Read the file into the buffer if it hasn't been loaded.
         * @note Getting a buffer without pre-load operation or implcit load flag leads to undefined behaviour.
         */
    virtual std::span<std::byte> get_buffer(bool implicit_load = true) = 0;

    template <class T>
    inline std::span<T> get_buffer(bool implicit_load = true) {
        std::span<std::byte> origin_buffer = get_buffer(implicit_load);
        T *buffer_ptr                      = reinterpret_cast<T *>(origin_buffer.data());
        const size_t buffer_size           = origin_buffer.size();
        return {buffer_ptr, buffer_size / sizeof(T)};
    }

    /*!
         * @brief Get the underlying implementaion method of FileLoader
         */
    virtual FileLoaderMethod get_method() const = 0;
};

/*!
     * @brief Factory Function: Build up a file loader
     */
std::unique_ptr<FileLoader> build_file_loader(const std::filesystem::path &file_path, FileLoaderMethod method);

} // namespace smart::storage
