#include "file_loader.hpp"

#include "file_loader/file_loader_bio.hpp"
#include "file_loader/file_loader_dio.hpp"
#include "file_loader/file_loader_mmap.hpp"
#include "file_loader/file_loader_uv.hpp"
#include "fmt/core.h"

#include <memory>
#include <unistd.h>

namespace smart::storage {

std::unique_ptr<FileLoader> build_file_loader(const std::filesystem::path &file_path, FileLoaderMethod method) {
    switch (method) {
    case FileLoaderMethod::BIO:
        return std::make_unique<FileLoaderBIO>(file_path);

    case FileLoaderMethod::MMap:
        return std::make_unique<FileLoaderMMap>(file_path);

    case FileLoaderMethod::DIO:
        return std::make_unique<FileLoaderDIO>(file_path);

    case FileLoaderMethod::UV:
        return std::make_unique<FileLoaderUV>(file_path);

    default:
        SMART_LOG_WARN("unknwon file loader method {}, fallback to bio", static_cast<int>(method));
        return std::make_unique<FileLoaderBIO>(file_path);
    }
}

} // namespace smart::storage
