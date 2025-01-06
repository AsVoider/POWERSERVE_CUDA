// Copyright 2024-2025 PowerServe Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <musa_runtime.h>
#include <musa.h>
#include <mublas.h>
#include <musa_fp16.h>
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define CUBLAS_COMPUTE_32F_FAST_16F MUBLAS_COMPUTE_32F_FAST_16F
#define CUBLAS_GEMM_DEFAULT MUBLAS_GEMM_DEFAULT
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP MUBLAS_GEMM_DEFAULT
#define CUBLAS_OP_N MUBLAS_OP_N
#define CUBLAS_OP_T MUBLAS_OP_T
#define CUBLAS_STATUS_SUCCESS MUBLAS_STATUS_SUCCESS
#define CUBLAS_TF32_TENSOR_OP_MATH MUBLAS_MATH_MODE_DEFAULT
#define CUDA_R_16F  MUSA_R_16F
#define CUDA_R_32F  MUSA_R_32F
#define cublasComputeType_t cudaDataType_t
#define cublasCreate mublasCreate
#define cublasDestroy mublasDestroy
#define cublasGemmEx mublasGemmEx
#define cublasGemmBatchedEx mublasGemmBatchedEx
#define cublasGemmStridedBatchedEx mublasGemmStridedBatchedEx
#define cublasHandle_t mublasHandle_t
#define cublasSetMathMode mublasSetMathMode
#define cublasSetStream mublasSetStream
#define cublasSgemm mublasSgemm
#define cublasStatus_t mublasStatus_t
#define cublasGetStatusString mublasStatus_to_string
#define cudaDataType_t musaDataType_t
#define cudaDeviceCanAccessPeer musaDeviceCanAccessPeer
#define cudaDeviceDisablePeerAccess musaDeviceDisablePeerAccess
#define cudaDeviceEnablePeerAccess musaDeviceEnablePeerAccess
#define cudaDeviceProp musaDeviceProp
#define cudaDeviceSynchronize musaDeviceSynchronize
#define cudaError_t musaError_t
#define cudaErrorPeerAccessAlreadyEnabled musaErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled musaErrorPeerAccessNotEnabled
#define cudaEventCreateWithFlags musaEventCreateWithFlags
#define cudaEventDisableTiming musaEventDisableTiming
#define cudaEventRecord musaEventRecord
#define cudaEventSynchronize musaEventSynchronize
#define cudaEvent_t musaEvent_t
#define cudaEventDestroy musaEventDestroy
#define cudaFree musaFree
#define cudaFreeHost musaFreeHost
#define cudaGetDevice musaGetDevice
#define cudaGetDeviceCount musaGetDeviceCount
#define cudaGetDeviceProperties musaGetDeviceProperties
#define cudaGetErrorString musaGetErrorString
#define cudaGetLastError musaGetLastError
#define cudaHostRegister musaHostRegister
#define cudaHostRegisterPortable musaHostRegisterPortable
#define cudaHostRegisterReadOnly musaHostRegisterReadOnly
#define cudaHostUnregister musaHostUnregister
#define cudaLaunchHostFunc musaLaunchHostFunc
#define cudaMalloc musaMalloc
#define cudaMallocHost musaMallocHost
#define cudaMemcpy musaMemcpy
#define cudaMemcpyAsync musaMemcpyAsync
#define cudaMemcpyPeerAsync musaMemcpyPeerAsync
#define cudaMemcpy2DAsync musaMemcpy2DAsync
#define cudaMemcpyDeviceToDevice musaMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost musaMemcpyDeviceToHost
#define cudaMemcpyHostToDevice musaMemcpyHostToDevice
#define cudaMemcpyKind musaMemcpyKind
#define cudaMemset musaMemset
#define cudaMemsetAsync musaMemsetAsync
#define cudaMemGetInfo musaMemGetInfo
#define cudaOccupancyMaxPotentialBlockSize musaOccupancyMaxPotentialBlockSize
#define cudaSetDevice musaSetDevice
#define cudaStreamCreateWithFlags musaStreamCreateWithFlags
#define cudaStreamDestroy musaStreamDestroy
#define cudaStreamFireAndForget musaStreamFireAndForget
#define cudaStreamNonBlocking musaStreamNonBlocking
#define cudaStreamPerThread musaStreamPerThread
#define cudaStreamSynchronize musaStreamSynchronize
#define cudaStreamWaitEvent musaStreamWaitEvent
#define cudaStream_t musaStream_t
#define cudaSuccess musaSuccess

// Additional mappings for MUSA virtual memory pool
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED MU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CU_MEM_ACCESS_FLAGS_PROT_READWRITE MU_MEM_ACCESS_FLAGS_PROT_READWRITE
#define CU_MEM_ALLOC_GRANULARITY_RECOMMENDED MU_MEM_ALLOC_GRANULARITY_RECOMMENDED
#define CU_MEM_ALLOCATION_TYPE_PINNED MU_MEM_ALLOCATION_TYPE_PINNED
#define CU_MEM_LOCATION_TYPE_DEVICE MU_MEM_LOCATION_TYPE_DEVICE
#define CUdevice MUdevice
#define CUdeviceptr MUdeviceptr
#define CUmemAccessDesc MUmemAccessDesc
#define CUmemAllocationProp MUmemAllocationProp
#define CUmemGenericAllocationHandle MUmemGenericAllocationHandle
#define cuDeviceGet muDeviceGet
#define cuDeviceGetAttribute muDeviceGetAttribute
#define cuMemAddressFree muMemAddressFree
#define cuMemAddressReserve muMemAddressReserve
#define cuMemCreate muMemCreate
#define cuMemGetAllocationGranularity muMemGetAllocationGranularity
#define cuMemMap muMemMap
#define cuMemRelease muMemRelease
#define cuMemSetAccess muMemSetAccess
#define cuMemUnmap muMemUnmap
#define cudaFuncAttributeMaxDynamicSharedMemorySize musaFuncAttributeMaxDynamicSharedMemorySize
#define cudaFuncSetAttribute musaFuncSetAttribute
#define cudaMemcpy3DPeerParms musaMemcpy3DPeerParms
#define make_cudaExtent make_musaExtent
#define make_cudaPitchedPtr make_musaPitchedPtr

// Additional mappings for MUSA graphs
#define CUDA_SUCCESS MUSA_SUCCESS
#define CUresult MUresult
#define cuGetErrorString muGetErrorString
#define cudaErrorGraphExecUpdateFailure musaErrorGraphExecUpdateFailure
#define cudaErrorInvalidDeviceFunction musaErrorInvalidDeviceFunction
#define cudaGraphDestroy musaGraphDestroy
#define cudaGraphExecDestroy musaGraphExecDestroy
#define cudaGraphExec_t musaGraphExec_t
#define cudaGraphExecUpdate musaGraphExecUpdate
#define cudaGraphExecUpdateResultInfo musaGraphExecUpdateResult
#define cudaGraphGetNodes musaGraphGetNodes
#define cudaGraphInstantiate musaGraphInstantiate
#define cudaGraphKernelNodeGetParams musaGraphKernelNodeGetParams
#define cudaGraphKernelNodeSetParams musaGraphKernelNodeSetParams
#define cudaGraphLaunch musaGraphLaunch
#define cudaGraphNodeGetType musaGraphNodeGetType
#define cudaGraphNode_t musaGraphNode_t
#define cudaGraphNodeType musaGraphNodeType
#define cudaGraphNodeTypeKernel musaGraphNodeTypeKernel
#define cudaGraph_t musaGraph_t
#define cudaKernelNodeParams musaKernelNodeParams
#define cudaStreamCaptureModeRelaxed musaStreamCaptureModeRelaxed
#define cudaStreamEndCapture musaStreamEndCapture
