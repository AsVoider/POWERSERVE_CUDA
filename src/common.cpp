#include "common.hpp"

namespace smart {

void get_memory_usage(const std::string &msg) {
#if defined(__ANDROID__)
    FILE *file = fopen("/proc/self/statm", "r");
    SMART_ASSERT(file != nullptr);

    size_t pages, resident, shared, text, lib, data, dt;
    fscanf(file, "%zu %zu %zu %zu %zu %zu %zu", &pages, &resident, &shared, &text, &lib, &data, &dt);

    fclose(file);

    long page_size = sysconf(_SC_PAGESIZE);
    size_t rss     = resident * page_size;
    size_t vms     = pages * page_size;

    fmt::println(stderr, "[{}] RSS: {} MB, VMS: {} MB", msg, rss / 1024 / 1024, vms / 1024 / 1024);
#elif defined(__linux__)
    FILE *file = fopen("/proc/self/status", "r");
    SMART_ASSERT(file != nullptr);
    char line[128];

    size_t rss = 0, vms = 0;
    while (fgets(line, sizeof(line), file)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line, "VmRSS: %zu", &rss);
            rss *= 1024; // Convert from KB to bytes
        }
        if (strncmp(line, "VmSize:", 7) == 0) {
            sscanf(line, "VmSize: %zu", &vms);
            vms *= 1024; // Convert from KB to bytes
        }
    }

    fclose(file);
    fmt::println(stderr, "[{}] RSS: {} MB, VMS: {} MB", msg, rss / 1024 / 1024, vms / 1024 / 1024);
#else
    SMART_UNUSED(msg);
#endif
}

} // namespace smart
