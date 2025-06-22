#include <cuda_runtime.h>

__global__ void convolution_1d(const float* A, const float* B, float* C, size_t N, size_t K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = (K - 1) / 2;

    if (i < N) {
        float sum = 0.0f;
        for (int j = 0; j < K; ++j) {
            int a_idx = i + j - r;  
            float a_val = (a_idx >= 0 && a_idx < N) ? A[a_idx] : 0.0f;  
            sum += a_val * B[j];
        }
        C[i] = sum;
    }
}


extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}