#include <cuda_runtime.h>

__global__ void relu(const float* input, float* output, size_t n, size_t m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n && col < m) {
        int index = row * m + col;
        float val = input[index];
        output[index] = val > 0 ? val : 0;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float* input, float* output, size_t n, size_t m) {    
    dim3 blockSize = (16,16);
    dim3 gridSize((m + blockSize.x - 1) / blockSize.x, 
              (n + blockSize.y - 1) / blockSize.y);
    relu<<<gridSize, blockSize>>>(input, output, n, m);
    cudaDeviceSynchronize();

}