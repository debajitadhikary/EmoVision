#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>

// Error checking macro
#define CHECK_CUDNN(call)                                           \
    {                                                                \
        cudnnStatus_t status = (call);                               \
        if (status != CUDNN_STATUS_SUCCESS) {                        \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(status) << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                            \
    }

#define CHECK_CUDA(call)                                            \
    {                                                                \
        cudaError_t err = (call);                                    \
        if (err != cudaSuccess) {                                    \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                     \
        }                                                            \
    }

int main() {
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Define tensor descriptors
    cudnnTensorDescriptor_t xDesc, yDesc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&yDesc));

    // Define tensor dimensions
    const int batch_size = 1;
    const int channels = 1;
    const int height = 5;
    const int width = 5;

    // Set tensor descriptors
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch_size, channels, height, width));

    // Define a convolution descriptor
    cudnnConvolutionDescriptor_t convDesc;
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

    // Define filter dimensions
    const int kernel_height = 3;
    const int kernel_width = 3;
    const int padding = 1;
    const int stride = 1;

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(convDesc, padding, padding, stride, stride, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(xDesc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(yDesc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(convDesc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    std::cout << "cuDNN is working correctly." << std::endl;

    return 0;
}
