#include "common.cuh"

int check_device_pointer(void *ptr) {
    cudaPointerAttributes attr;
    auto err{cudaPointerGetAttributes(&attr, ptr)};

    if (err == cudaSuccess) {
        if (attr.type == cudaMemoryTypeDevice) {
            return attr.device;
        } else if (attr.type == cudaMemoryTypeHost) {
            return -1;
        } else {
            // TODO:
            exit(1);
            // return -2;
        }
    } else {
        // TODO:
        exit(1);
    }
}

template <ggml_type T_type>
auto print_tensor_to_file(std::string file_name, std::string tensor_name, const ggml_tensor *tensor, const size_t n_rows, const size_t n_cols) -> void {
    FILE *file{fopen(file_name.c_str(), "a+")};
    if (not file) {
        printf("Cannot open file: %s\n", file_name.c_str());
        return;
    }

    GGML_ASSERT(tensor->data not_eq nullptr);

    if constexpr (T_type == GGML_TYPE_F32) {
        size_t size{n_rows * n_cols * sizeof(float)};
        float *buffer_host{new float[n_rows * n_cols]};
        cudaMemcpy(buffer_host, tensor->data, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (size_t i{0}; i < n_rows; ++i) {
            for (size_t j{0}; j < n_cols; ++j) {
                fprintf(file, "%f ", buffer_host[i * n_cols + j]);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");

        fclose(file);
        return;
    }

    if constexpr (T_type == GGML_TYPE_F16) {
        size_t size{n_rows * n_cols * sizeof(half)};
        half *buffer_host{new half[n_rows * n_cols]};
        cudaMemcpy(buffer_host, tensor->data, size, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (size_t i{0}; i < n_rows; ++i) {
            for (size_t j{0}; j < n_cols; ++j) {
                float data_to_print{static_cast<float>(buffer_host[i * n_cols + j])};
                fprintf(file, "%f ", data_to_print);
            }
            fprintf(file, "\n");
        }
        fprintf(file, "\n\n");

        fclose(file);
        return;
    }

    if constexpr (T_type == GGML_TYPE_Q4_0) {
        fclose(file);
        return;
    }

    if constexpr (T_type == GGML_TYPE_Q8_0) {
        fclose(file);
        return;
    }
}