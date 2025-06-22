#include <cuda_runtime.h>

__global__ void avgPool(const float* input, int kernel_size, int stride, int padding, size_t output_size, float* output, size_t H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < output_size) {
        output[i] = 0;
        int m = 0;
        while (m <= kernel_size - 1) {
            int input_idx = stride*i+m-padding;
            if (input_idx >= 0 && input_idx < H) {
                output[i] = output[i] + input[input_idx];
            }
            m = m + 1;
        }
        output[i] = output[i] / kernel_size;
    }
}

extern "C" void solution(const float* input, int kernel_size, int stride, int padding, float* output, size_t H) {    
    size_t output_size = floor((H+2*padding-kernel_size)/stride + 1);

    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    avgPool<<<blocksPerGrid, threadsPerBlock>>>(input, kernel_size, stride, padding, output_size, output, H);
    cudaDeviceSynchronize();

}