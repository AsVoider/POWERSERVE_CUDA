#pragma once

#include "common/logger.hpp"

#include <unistd.h>

namespace smart::storage {

///
/// RAII Wrapper for file descriptor
///
struct FileHandle {
public:
    static constexpr int INVALID_FILE_HANDLE = -1;

public:
    /// File descritor
    int m_fd = INVALID_FILE_HANDLE;

public:
    FileHandle() = default;

    FileHandle(const int fd) : m_fd(fd) {}

    ~FileHandle() noexcept {
        reset();
    }

    FileHandle(const FileHandle &other) = delete;

    FileHandle(FileHandle &&other) noexcept : m_fd(other.m_fd) {
        other.m_fd = INVALID_FILE_HANDLE;
    }

    FileHandle &operator=(const FileHandle &other) = delete;

    FileHandle &operator=(FileHandle &&other) noexcept {
        if (this != &other) {
            reset(other.m_fd);
            other.m_fd = INVALID_FILE_HANDLE;
        }
        return *this;
    }

public:
    /*!
         * @brief Reset file descriptor with a new one
         * @param[in] new_fd New file descirptor assigned to the handle
         * @note If the file handle has already been assigned, it will close the old one and hold the new one.
         */
    void reset(const int new_fd = INVALID_FILE_HANDLE) {
        if (m_fd != INVALID_FILE_HANDLE) {
            const int ret = close(m_fd);
            SMART_ASSERT(ret != -1, "failed to close file {}", m_fd);
        }
        m_fd = new_fd;
    }

    /*!
         * @brief Reset file descriptor directly
         * @note This may lead to resources leak. Only use it after handling the file descriptor properly
         */
    void unsafe_reset() {
        m_fd = INVALID_FILE_HANDLE;
    }
};

} // namespace smart::storage
