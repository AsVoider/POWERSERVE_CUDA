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

void get_system_temperature(const std::string &msg) {
    Path thermal_zone_path{"/sys/class/thermal/thermal_zone0/temp"};
    std::ifstream thermal_zone_file(thermal_zone_path);
    if (!thermal_zone_file.is_open()) {
        fmt::println(stderr, "Failed to open: {}, Please check file or permission", thermal_zone_path);
        return;
    }

    long long temperature_val;
    thermal_zone_file >> temperature_val;
    thermal_zone_file.close();

    double celsius = 0;
#if defined(__ANDROID__)
    celsius = temperature_val / 100.0;
#elif defined(__linux__)
    celsius = temperature_val / 1000.0;
#endif

    fmt::println("[{}] Current CPU temperature: {}°C", msg, celsius);
    return;
}

} // namespace smart
