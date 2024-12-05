#include "qnn.hpp"

#include "HTP/QnnHtpGraph.h"
#include "HTP/QnnHtpMem.h"
#include "HTP/QnnHtpSystemContext.h"
#include "qnn_type_macros.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace smart::qnn {

static void log_callback(const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
    SMART_UNUSED(level);
    SMART_UNUSED(timestamp);
    vprintf(fmt, args);
}

static auto format_qnn_version(Qnn_Version_t version) -> std::string {
    return fmt::format("{}.{}.{}", version.major, version.minor, version.patch);
}

Library::~Library() {
    destroy_logger();
    uv_dlclose(&m_lib_system);
    uv_dlclose(&m_lib_backend);
    uv_dlclose(&m_lib_rpc);
}

void Library::initialize(const Path &lib_backend_path, const Path &lib_system_path, const Path &lib_rpc_path) {
    open_qnn_backend_library(lib_backend_path);
    open_qnn_system_library(lib_system_path);
    open_rpc_library(lib_rpc_path);
    create_logger();
}

void Library::open_qnn_backend_library(const Path &lib_backend_path) {
    int ret = uv_dlopen(lib_backend_path.c_str(), &m_lib_backend);
    SMART_ASSERT(ret == 0);

    typedef Qnn_ErrorHandle_t (*QnnInterfaceGetProvidersFn_t)(
        const QnnInterface_t ***providerList, uint32_t *numProviders
    );

    QnnInterfaceGetProvidersFn_t get_interface_providers;
    ret = uv_dlsym(&m_lib_backend, "QnnInterface_getProviders", (void **)&get_interface_providers);
    SMART_ASSERT(ret == 0);

    const QnnInterface_t **interface_providers;
    uint32_t n_providers = 0;
    ret                  = get_interface_providers(&interface_providers, &n_providers);
    SMART_ASSERT(ret == QNN_SUCCESS);
    SMART_ASSERT(n_providers > 0);

    bool found = false;
    for (size_t i = 0; i < n_providers; i++) {
        auto api_version = interface_providers[i]->apiVersion.coreApiVersion;
        if (QNN_API_VERSION_MAJOR == api_version.major && QNN_API_VERSION_MINOR <= api_version.minor) {
            found         = true;
            m_qnn_backend = interface_providers[i]->QNN_INTERFACE_VER_NAME;
            break;
        }
    }
    SMART_ASSERT(found);

    Qnn_ApiVersion_t api_version;
    ret = m_qnn_backend.backendGetApiVersion(&api_version);
    SMART_ASSERT(ret == QNN_SUCCESS);
    fmt::println("QNN core API version: {}", format_qnn_version(api_version.coreApiVersion));
    fmt::println("QNN backend API version: {}", format_qnn_version(api_version.backendApiVersion));
}

void Library::open_qnn_system_library(const Path &lib_system_path) {
    int ret = uv_dlopen(lib_system_path.c_str(), &m_lib_system);
    SMART_ASSERT(ret == 0);

    typedef Qnn_ErrorHandle_t (*QnnSystemInterfaceGetProvidersFn_t)(
        const QnnSystemInterface_t ***providerList, uint32_t *numProviders
    );

    QnnSystemInterfaceGetProvidersFn_t get_system_inferface_providers;
    ret = uv_dlsym(&m_lib_system, "QnnSystemInterface_getProviders", (void **)&get_system_inferface_providers);
    SMART_ASSERT(ret == 0);

    const QnnSystemInterface_t **system_interface_providers;
    uint32_t n_providers = 0;
    ret                  = get_system_inferface_providers(&system_interface_providers, &n_providers);
    SMART_ASSERT(ret == QNN_SUCCESS);
    SMART_ASSERT(n_providers > 0);

    bool found = false;
    for (size_t i = 0; i < n_providers; i++) {
        auto api_version = system_interface_providers[i]->systemApiVersion;
        if (QNN_SYSTEM_API_VERSION_MAJOR == api_version.major && QNN_SYSTEM_API_VERSION_MINOR <= api_version.minor) {
            found = true;
            fmt::println("QNN system API verion: {}", format_qnn_version(api_version));
            m_qnn_system = system_interface_providers[i]->QNN_SYSTEM_INTERFACE_VER_NAME;
            break;
        }
    }
    SMART_ASSERT(found);
}

void Library::open_rpc_library(const Path &lib_rpc_path) {
    int ret = uv_dlopen(lib_rpc_path.c_str(), &m_lib_rpc);
    SMART_ASSERT(ret == 0);

    ret = uv_dlsym(&m_lib_rpc, "rpcmem_alloc", (void **)&m_rpc.rpcmem_alloc);
    SMART_ASSERT(ret == 0);

    ret = uv_dlsym(&m_lib_rpc, "rpcmem_free", (void **)&m_rpc.rpcmem_free);
    SMART_ASSERT(ret == 0);

    ret = uv_dlsym(&m_lib_rpc, "rpcmem_to_fd", (void **)&m_rpc.rpcmem_to_fd);
    SMART_ASSERT(ret == 0);
}

void Library::print_info() {
    auto print_property = [&](const std::string &name, QnnProperty_Key_t property) {
        auto ret = m_qnn_backend.propertyHasCapability(property);

        const char *status = "Unknown";
        if (ret == QNN_PROPERTY_SUPPORTED) {
            status = "Yes";
        } else if (ret == QNN_PROPERTY_NOT_SUPPORTED) {
            status = "No";
        }

        fmt::println("- {}: {}", name, status);
    };

    fmt::println("QNN backend properties:");
    print_property("Create context from binary list", QNN_PROPERTY_CONTEXT_SUPPORT_CREATE_FROM_BINARY_LIST_ASYNC);
    print_property("Dynamic batch", QNN_PROPERTY_GRAPH_SUPPORT_BATCH_MULTIPLE);
    print_property("Early termination", QNN_PROPERTY_GRAPH_SUPPORT_EARLY_TERMINATION);
    print_property("Dynamic dimensions", QNN_PROPERTY_TENSOR_SUPPORT_DYNAMIC_DIMENSIONS);
    print_property("Blockwise quantization", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCK);
    print_property(
        "Blockwise quantization with expansion", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_BLOCKWISE_EXPANSION
    );
    print_property("Vector quantization", QNN_PROPERTY_TENSOR_SUPPORT_QUANTIZATION_ENCODING_VECTOR);
    print_property("Tensor sparsity", QNN_PROPERTY_TENSOR_SUPPORT_SPARSITY);
    print_property("Updateable application tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_APP_TENSORS);
    print_property("Updateable native tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_NATIVE_TENSORS);
    print_property("Updateable static tensor", QNN_PROPERTY_TENSOR_SUPPORT_UPDATEABLE_STATIC_TENSORS);
}

void Library::create_logger() {
    auto ret = m_qnn_backend.logCreate(log_callback, m_log_level, &m_logger);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

void Library::destroy_logger() {
    if (m_logger) {
        auto ret = m_qnn_backend.logFree(m_logger);
        SMART_ASSERT(ret == QNN_SUCCESS);
        m_logger = nullptr;
    }
}

Library lib;

Backend::Backend() {
    auto ret = lib.m_qnn_backend.backendCreate(lib.m_logger, nullptr, &m_handle);
    SMART_ASSERT(ret == QNN_SUCCESS);

    ret = lib.m_qnn_backend.deviceCreate(lib.m_logger, nullptr, &m_device);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

Backend::~Backend() {
    auto ret = lib.m_qnn_backend.deviceFree(m_device);
    SMART_ASSERT(ret == QNN_SUCCESS);

    ret = lib.m_qnn_backend.backendFree(m_handle);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

void Backend::print_info() {
    const QnnDevice_PlatformInfo_t *platform_info_ptr;
    auto ret = lib.m_qnn_backend.deviceGetInfo(m_device, &platform_info_ptr);
    SMART_ASSERT(ret == QNN_SUCCESS);
    SMART_ASSERT(platform_info_ptr->version == QNN_DEVICE_PLATFORM_INFO_VERSION_1);

    auto &platform_info = platform_info_ptr->v1;

    fmt::println("Hardware device infomation:");
    for (size_t i = 0; i < platform_info.numHwDevices; i++) {
        auto &hw_info_struct = platform_info.hwDevices[i];
        SMART_ASSERT(hw_info_struct.version == QNN_DEVICE_HARDWARE_DEVICE_INFO_VERSION_1);

        auto &hw_info = hw_info_struct.v1;
        fmt::println(
            "[{}] id={}, type={}, num_cores={}, ext_type={}",
            i,
            hw_info.deviceId,
            hw_info.deviceType,
            hw_info.numCores,
            (int)hw_info.deviceInfoExtension->devType
        );

        for (size_t j = 0; j < hw_info.numCores; j++) {
            auto &core_info_struct = hw_info.cores[j];
            SMART_ASSERT(core_info_struct.version == QNN_DEVICE_CORE_INFO_VERSION_1);

            auto &core_info = core_info_struct.v1;
            fmt::println("[{}] core[{}]: id={}, type={}", i, j, core_info.coreId, core_info.coreType);
        }

        if (hw_info.deviceInfoExtension->devType == QNN_HTP_DEVICE_TYPE_ON_CHIP) {
            auto &on_chip_info = hw_info.deviceInfoExtension->onChipDevice;
            fmt::println(
                "[{}] on_chip: soc={}, arch={}, dlbc={}, signed_pd={}, vtcm_size={}",
                i,
                on_chip_info.socModel,
                (int)on_chip_info.arch,
                on_chip_info.dlbcSupport,
                on_chip_info.signedPdSupport,
                on_chip_info.vtcmSize
            );
        }
    }

    ret = lib.m_qnn_backend.deviceFreePlatformInfo(lib.m_logger, platform_info_ptr);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

HTPDevice::HTPDevice(uint32_t device_id, uint32_t core_id) : m_device_id(device_id), m_core_id(core_id) {
    auto ret = lib.m_qnn_backend.deviceGetInfrastructure(&m_infra);
    SMART_ASSERT(ret == QNN_SUCCESS);

    m_htp_infra  = (QnnHtpDevice_Infrastructure_t *)m_infra;
    m_perf_infra = m_htp_infra->perfInfra;
    ret          = m_perf_infra.createPowerConfigId(device_id, core_id, &m_power_config_id);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

HTPDevice::~HTPDevice() {
    auto ret = m_perf_infra.destroyPowerConfigId(m_power_config_id);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

void HTPDevice::set_memory_grow_size(size_t size) {
    QnnHtpPerfInfrastructure_MemoryConfig_t grow_size_config = {
        .option            = QNN_HTP_PERF_INFRASTRUCTURE_MEMORY_CONFIGOPTION_GROW_SIZE,
        .memGrowSizeConfig = (uint32_t)size,
    };

    const QnnHtpPerfInfrastructure_MemoryConfig_t *memory_config[] = {
        &grow_size_config,
        nullptr,
    };
    auto ret = m_perf_infra.setMemoryConfig(m_device_id, m_core_id, memory_config);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

void HTPDevice::enter_performance_mode() {
    QnnHtpPerfInfrastructure_PowerConfig_t dcvs_v3_config = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3,
        .dcvsV3Config =
            {
                .contextId = m_power_config_id,

                .setDcvsEnable = 1,
                .dcvsEnable    = 0,

                .powerMode = QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE,

                .setSleepLatency = 1,
                .sleepLatency    = 40,

                .setSleepDisable = 1,
                .sleepDisable    = 1,

                .setBusParams           = 1,
                .busVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .busVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .busVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,

                .setCoreParams           = 1,
                .coreVoltageCornerMin    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .coreVoltageCornerTarget = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
                .coreVoltageCornerMax    = DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER,
            },
    };

    QnnHtpPerfInfrastructure_PowerConfig_t hmx_config = {
        .option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_V2,
        .hmxV2Config =
            {
                .hmxPickDefault         = 0,
                .hmxVoltageCornerMin    = DCVS_EXP_VCORNER_MAX,
                .hmxVoltageCornerTarget = DCVS_EXP_VCORNER_MAX,
                .hmxVoltageCornerMax    = DCVS_EXP_VCORNER_MAX,
                .hmxPerfMode            = QNN_HTP_PERF_INFRASTRUCTURE_CLK_PERF_HIGH,
            },
    };

    QnnHtpPerfInfrastructure_PowerConfig_t rpc_ctrl_config = {
        .option                  = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY,
        .rpcControlLatencyConfig = 100,
    };

    QnnHtpPerfInfrastructure_PowerConfig_t rpc_poll_config = {
        .option               = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME,
        .rpcPollingTimeConfig = 9999,
    };

    const QnnHtpPerfInfrastructure_PowerConfig_t *power_configs[] = {
        &dcvs_v3_config,
        &hmx_config,
        &rpc_ctrl_config,
        &rpc_poll_config,
        nullptr,
    };
    auto ret = m_perf_infra.setPowerConfig(m_power_config_id, power_configs);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

ContextGroup::ContextGroup(size_t buffer_size) : m_buffer_size(buffer_size) {}

auto ContextGroup::get_config() const -> QnnHtpContext_CustomConfig_t {
    return QnnHtpContext_CustomConfig_t{
        .option = QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS,
        .groupRegistration =
            {
                .firstGroupHandle   = m_first_handle,
                .maxSpillFillBuffer = m_buffer_size,
            },
    };
}

void ContextGroup::add_context_handle(Qnn_ContextHandle_t handle) {
    if (!m_first_handle) {
        m_first_handle = handle;
    }
}

Context::Context(Backend &backend, const Path &binary_file_path, ContextGroup *group) {
    SMART_ASSERT(std::filesystem::exists(binary_file_path));

    std::ifstream file(binary_file_path, std::ios::binary | std::ios::ate);
    SMART_ASSERT(file.is_open());

    auto pos = file.tellg();
    SMART_ASSERT(pos >= 0);
    std::vector<char> binary_data(pos);
    file.seekg(0, std::ios::beg);
    file.read(binary_data.data(), pos);
    SMART_ASSERT(file.tellg() == pos);

    std::vector<const QnnContext_Config_t *> context_configs;

    QnnHtpContext_CustomConfig_t htp_group_config;
    QnnContext_Config_t group_config = {
        .option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_group_config,
    };

    if (group) {
        htp_group_config = group->get_config();
        context_configs.push_back(&group_config);
    }

    QnnHtpContext_CustomConfig_t htp_io_estimation_config = {
        .option          = QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION,
        .ioMemEstimation = true,
    };
    QnnContext_Config_t io_estimation_config = {
        .option       = QNN_CONTEXT_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_io_estimation_config,
    };
    context_configs.push_back(&io_estimation_config);

    context_configs.push_back(nullptr);

    auto ret = lib.m_qnn_backend.contextCreateFromBinary(
        backend.m_handle,
        backend.m_device,
        context_configs.data(),
        binary_data.data(),
        binary_data.size(),
        &m_handle,
        nullptr
    );
    SMART_ASSERT(ret == QNN_SUCCESS);

    if (group) {
        group->add_context_handle(m_handle);
    }

    ret = lib.m_qnn_system.systemContextCreate(&m_system_context);
    SMART_ASSERT(ret == QNN_SUCCESS);

    Qnn_ContextBinarySize_t binary_info_size = 0;
    ret                                      = lib.m_qnn_system.systemContextGetBinaryInfo(
        m_system_context, binary_data.data(), binary_data.size(), &m_binary_info, &binary_info_size
    );
    SMART_ASSERT(ret == QNN_SUCCESS);
    SMART_ASSERT(m_binary_info->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1);
}

Context::~Context() {
    free_system_context();
    auto ret = lib.m_qnn_backend.contextFree(m_handle, nullptr);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

void Context::print_info() {
    SMART_ASSERT(m_binary_info);

    auto &info = m_binary_info->contextBinaryInfoV1;

    auto hw_blob_info_ptr = (QnnHtpSystemContext_HwBlobInfo_t *)info.hwInfoBlob;
    SMART_ASSERT(hw_blob_info_ptr->version == QNN_SYSTEM_CONTEXT_HTP_HW_INFO_BLOB_VERSION_V1);
    auto &hw_blob_info = hw_blob_info_ptr->contextBinaryHwInfoBlobV1_t;

    fmt::println("Context core API version: {}", format_qnn_version(info.coreApiVersion));
    fmt::println("Context backend API version: {}", format_qnn_version(info.backendApiVersion));
    fmt::println("Context blob version: {}", format_qnn_version(info.contextBlobVersion));
    fmt::println("Number of graphs: {}", info.numGraphs);
    fmt::println("Spill-fill buffer size: {:.3f} MiB", hw_blob_info.spillFillBufferSize / 1024.0 / 1024);
}

void Context::free_system_context() {
    if (m_system_context) {
        auto ret = lib.m_qnn_system.systemContextFree(m_system_context);
        SMART_ASSERT(ret == QNN_SUCCESS);
    }

    m_binary_info    = nullptr;
    m_system_context = nullptr;
}

SharedBufferAllocator::SharedBufferAllocator(size_t _size) : m_size(_size) {
    m_data = lib.m_rpc.rpcmem_alloc(lib.m_rpc.RPCMEM_HEAP_ID_SYSTEM, lib.m_rpc.RPCMEM_DEFAULT_FLAGS, m_size);
    SMART_ASSERT(m_data);

    m_fd = lib.m_rpc.rpcmem_to_fd(m_data);
    SMART_ASSERT(m_fd != -1);
}

SharedBufferAllocator::~SharedBufferAllocator() {
    lib.m_rpc.rpcmem_free(m_data);
}

auto SharedBufferAllocator::unallocated_size() const -> size_t {
    return m_size - m_offset;
}

SharedBuffer::SharedBuffer(Context &context, SharedBufferAllocator &allocator, QNNDataType type, size_t n_elements) :
    m_type(type) {
    m_size = type_size(type) * n_elements;

    SMART_ASSERT(allocator.m_offset + m_size <= allocator.m_size);
    m_data = (char *)allocator.m_data + allocator.m_offset;

    QnnMemHtp_Descriptor_t htp_mem_desc = {
        .type = QNN_HTP_MEM_SHARED_BUFFER,
        .size = allocator.m_size, // NOTE: It's the total size of the shared buffer allocator
        .sharedBufferConfig =
            (QnnHtpMem_SharedBufferConfig_t){
                .fd     = allocator.m_fd,
                .offset = allocator.m_offset,
            },
    };

    uint32_t shape[1]            = {(uint32_t)n_elements};
    Qnn_MemDescriptor_t mem_desc = {
        .memShape =
            {
                .numDim      = 1,
                .dimSize     = shape,
                .shapeConfig = nullptr,
            },
        .dataType   = type,
        .memType    = QNN_MEM_TYPE_CUSTOM,
        .customInfo = &htp_mem_desc,
    };

    auto ret = lib.m_qnn_backend.memRegister(context.m_handle, &mem_desc, 1, &m_handle);
    if (ret != QNN_SUCCESS) {
        throw std::runtime_error("Shared Buffer Error");
    }

    allocator.m_offset += m_size;
}

SharedBuffer::~SharedBuffer() {
    auto ret = lib.m_qnn_backend.memDeRegister(&m_handle, 1);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

void SharedBuffer::memset(int byte) {
    ::memset(m_data, byte, m_size);
}

static void deep_copy_tensor(Qnn_Tensor_t &dst, const Qnn_Tensor_t &src) {
    dst = QNN_TENSOR_INIT;

    // set tensor.version before using QNN_TENSOR_SET macros, as they require the version to be set
    // to correctly assign values
    dst.version = src.version;

    const char *tensorName = QNN_TENSOR_GET_NAME(src);
    if (!tensorName) {
        QNN_TENSOR_SET_NAME(dst, nullptr);
    } else {
        QNN_TENSOR_SET_NAME(dst, strdup(tensorName));
    }

    QNN_TENSOR_SET_ID(dst, QNN_TENSOR_GET_ID(src));
    QNN_TENSOR_SET_TYPE(dst, QNN_TENSOR_GET_TYPE(src));
    QNN_TENSOR_SET_DATA_FORMAT(dst, QNN_TENSOR_GET_DATA_FORMAT(src));
    QNN_TENSOR_SET_DATA_TYPE(dst, QNN_TENSOR_GET_DATA_TYPE(src));

    Qnn_QuantizeParams_t qParams = QNN_QUANTIZE_PARAMS_INIT;
    qParams.encodingDefinition   = QNN_TENSOR_GET_QUANT_PARAMS(src).encodingDefinition;
    qParams.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;

    if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) {
        qParams.quantizationEncoding = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.scaleOffsetEncoding  = QNN_TENSOR_GET_QUANT_PARAMS(src).scaleOffsetEncoding;
    } else if (QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding == QNN_QUANTIZATION_ENCODING_AXIS_SCALE_OFFSET) {
        qParams.quantizationEncoding         = QNN_TENSOR_GET_QUANT_PARAMS(src).quantizationEncoding;
        qParams.axisScaleOffsetEncoding.axis = QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.axis;
        qParams.axisScaleOffsetEncoding.numScaleOffsets =
            QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;

        if (QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets > 0) {
            qParams.axisScaleOffsetEncoding.scaleOffset = (Qnn_ScaleOffset_t *)malloc(
                QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets * sizeof(Qnn_ScaleOffset_t)
            );

            if (qParams.axisScaleOffsetEncoding.scaleOffset) {
                for (size_t idx = 0; idx < QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.numScaleOffsets;
                     idx++) {
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].scale =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].scale;
                    qParams.axisScaleOffsetEncoding.scaleOffset[idx].offset =
                        QNN_TENSOR_GET_QUANT_PARAMS(src).axisScaleOffsetEncoding.scaleOffset[idx].offset;
                }
            }
        }
    }

    QNN_TENSOR_SET_QUANT_PARAMS(dst, qParams);
    QNN_TENSOR_SET_RANK(dst, QNN_TENSOR_GET_RANK(src));
    QNN_TENSOR_SET_DIMENSIONS(dst, nullptr);

    if (QNN_TENSOR_GET_RANK(src) > 0) {
        QNN_TENSOR_SET_DIMENSIONS(dst, (uint32_t *)malloc(QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)));

        if (QNN_TENSOR_GET_DIMENSIONS(dst)) {
            memcpy(
                QNN_TENSOR_GET_DIMENSIONS(dst),
                QNN_TENSOR_GET_DIMENSIONS(src),
                QNN_TENSOR_GET_RANK(src) * sizeof(uint32_t)
            );
        }

        // I think the original code is wrong...
        SMART_ASSERT(!QNN_TENSOR_GET_IS_DYNAMIC_DIMENSIONS(src));
    }

    QNN_TENSOR_SET_SPARSE_PARAMS(dst, QNN_TENSOR_GET_SPARSE_PARAMS(src));
}

std::unordered_map<Tensor *, void *> buffer_map{};

Tensor::Tensor(const Qnn_Tensor_t &source) {
    deep_copy_tensor(m_tensor, source);
    SMART_ASSERT(QNN_TENSOR_GET_MEM_TYPE(m_tensor) == QNN_TENSORMEMTYPE_UNDEFINED);
}

Tensor::~Tensor() {
    switch (QNN_TENSOR_GET_MEM_TYPE(m_tensor)) {
    case QNN_TENSORMEMTYPE_RAW: {
        free(QNN_TENSOR_GET_CLIENT_BUF(m_tensor).data);
    } break;

    case QNN_TENSORMEMTYPE_MEMHANDLE: // Released by SharedBufferAllocator
    case QNN_TENSORMEMTYPE_UNDEFINED:
        break;

    default:
        SMART_ASSERT(false);
    }
    QNN_TENSOR_SET_MEM_TYPE(m_tensor, QNN_TENSORMEMTYPE_UNDEFINED);

    free(QNN_TENSOR_GET_DIMENSIONS(m_tensor));
    free((void *)QNN_TENSOR_GET_NAME(m_tensor));
}

auto Tensor::name() const -> std::string {
    return QNN_TENSOR_GET_NAME(m_tensor);
}

size_t Tensor::n_elements() const {
    size_t n_elements = 1;
    for (size_t i = 0; i < QNN_TENSOR_GET_RANK(m_tensor); i++) {
        n_elements *= QNN_TENSOR_GET_DIMENSIONS(m_tensor)[i];
    }
    return n_elements;
}

auto Tensor::type() const -> QNNDataType {
    return QNN_TENSOR_GET_DATA_TYPE(m_tensor);
}

size_t Tensor::size() const {
    return n_elements() * type_size(type());
}

auto Tensor::shape() const -> std::vector<size_t> {
    std::vector<size_t> shape(QNN_TENSOR_GET_RANK(m_tensor));
    for (size_t i = 0; i < shape.size(); i++) {
        shape[i] = QNN_TENSOR_GET_DIMENSIONS(m_tensor)[i];
    }
    return shape;
}

void Tensor::setup_normal_buffer() {
    Qnn_ClientBuffer_t buffer = QNN_CLIENT_BUFFER_INIT;
    buffer.dataSize           = size();
    buffer.data               = malloc(buffer.dataSize);
    QNN_TENSOR_SET_MEM_TYPE(m_tensor, QNN_TENSORMEMTYPE_RAW);
    QNN_TENSOR_SET_CLIENT_BUF(m_tensor, buffer);
}

void Tensor::setup_shared_buffer(SharedBuffer &buffer) {
    QNN_TENSOR_SET_MEM_TYPE(m_tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
    QNN_TENSOR_SET_MEM_HANDLE(m_tensor, buffer.m_handle);
    buffer_map.emplace(this, buffer.m_data);
}

auto Tensor::data() -> void * {
    switch (QNN_TENSOR_GET_MEM_TYPE(m_tensor)) {
    case QNN_TENSORMEMTYPE_RAW:
        return QNN_TENSOR_GET_CLIENT_BUF(m_tensor).data;

    case QNN_TENSORMEMTYPE_MEMHANDLE: // User should fill the shared buffer directly
    default:
        SMART_ASSERT(false);
    }
}

int Tensor::quantization_offset() const {
    return QNN_TENSOR_GET_QUANT_PARAMS(m_tensor).scaleOffsetEncoding.offset;
}

float Tensor::quantization_scale() const {
    return QNN_TENSOR_GET_QUANT_PARAMS(m_tensor).scaleOffsetEncoding.scale;
}

auto Tensor::check(const std::vector<size_t> &shape, Qnn_DataType_t datatype) -> Tensor * {
    SMART_ASSERT(this->shape() == shape);
    SMART_ASSERT(this->type() == datatype);
    return this;
}

void Tensor::print() {
    if (type() == QNN_DATATYPE_FLOAT_32) {
        auto buf = (const float *)buffer_map.at(this);
        for (size_t i = 0; i < n_elements(); i++) {
            fmt::println(stderr, "{}", buf[i]);
        }
    } else if (type() == QNN_DATATYPE_FLOAT_16) {
        auto buf = (const __fp16 *)buffer_map.at(this);
        for (size_t i = 0; i < n_elements(); i++) {
            fmt::println(stderr, "{}", (float)buf[i]);
        }
    }
}

Graph::Graph(Context &context, const std::string &name) {
    auto &info = context.m_binary_info->contextBinaryInfoV1;

    const QnnSystemContext_GraphInfoV1_t *graph_info = nullptr;
    for (size_t i = 0; i < info.numGraphs; i++) {
        const auto *current_graph = &info.graphs[i];
        SMART_ASSERT(current_graph->version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1);
        if (current_graph->graphInfoV1.graphName == name) {
            graph_info = &current_graph->graphInfoV1;
            break;
        }
    }
    SMART_ASSERT(graph_info);

    size_t n_inputs = graph_info->numGraphInputs;
    m_inputs.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; i++) {
        m_inputs.emplace_back(graph_info->graphInputs[i]);
    }

    size_t n_outputs = graph_info->numGraphOutputs;
    m_outputs.reserve(n_outputs);
    for (size_t i = 0; i < n_outputs; i++) {
        m_outputs.emplace_back(graph_info->graphOutputs[i]);
    }

    auto ret = lib.m_qnn_backend.graphRetrieve(context.m_handle, name.c_str(), &m_handle);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

auto Graph::get_tensor(const std::string &name, bool required) -> Tensor * {
    for (auto &t : m_inputs) {
        if (t.name() == name) {
            return &t;
        }
    }

    for (auto &t : m_outputs) {
        if (t.name() == name) {
            return &t;
        }
    }

    if (required) {
        fmt::println(stderr, "Cannot find tensor with name \"{}\"", name);
        SMART_ASSERT(false);
    } else {
        return nullptr;
    }
}

bool Graph::has_tensor(const std::string &name) {
    return get_tensor(name, false) != nullptr;
}

void Graph::set_n_hvx_threads(size_t n_threads) {
    QnnHtpGraph_CustomConfig_t htp_hvx_thread_config = {
        .option        = QNN_HTP_GRAPH_CONFIG_OPTION_NUM_HVX_THREADS,
        .numHvxThreads = n_threads,
    };

    QnnGraph_Config_t hvx_thread_config = {
        .option       = QNN_GRAPH_CONFIG_OPTION_CUSTOM,
        .customConfig = &htp_hvx_thread_config,
    };

    const QnnGraph_Config_t *graph_configs[] = {&hvx_thread_config, nullptr};
    auto ret                                 = lib.m_qnn_backend.graphSetConfig(m_handle, graph_configs);
    SMART_ASSERT(ret == QNN_SUCCESS);
}

void Graph::execute() {
    auto ret = lib.m_qnn_backend.graphExecute(
        m_handle,
        (const Qnn_Tensor_t *)m_inputs.data(),
        m_inputs.size(),
        (Qnn_Tensor_t *)m_outputs.data(),
        m_outputs.size(),
        nullptr,
        nullptr
    );
    SMART_ASSERT(ret == QNN_SUCCESS);
}

Session::Session(const Path &libs_folder) {
    m_count = 0;
    uv_os_setenv("ADSP_LIBRARY_PATH", libs_folder.c_str());
    lib.initialize(libs_folder / "libQnnHtp.so", libs_folder / "libQnnSystem.so");
    lib.print_info();

    m_backend = std::make_unique<qnn::Backend>();
    m_backend->print_info();

    m_htp_device = std::make_unique<HTPDevice>();
    m_htp_device->set_memory_grow_size();
    m_htp_device->enter_performance_mode();

    m_group = std::make_unique<ContextGroup>(10 * 1024 * 1024);
}

ContextBinary::ContextBinary(Backend &backend, const Path &path) {
    m_context = std::make_unique<Context>(backend, path, nullptr);
}

} // namespace smart::qnn
