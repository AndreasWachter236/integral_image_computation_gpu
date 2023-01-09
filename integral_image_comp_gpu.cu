#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include "image.h"

const int blockSize = 1024;
const int gridSize = 64;

int const image_width = 10000;
int const image_height = 100000;

using namespace std;

template <typename T>
void createRandomImage(Image<T>& rndImage)
{
    for (int j = 0; j < rndImage.pixels; j++)
    {
        rndImage.data[j] = 1;// (T) rand() % 256);
    }
}


template <typename T>
T sum(Image<T>& iImage, const size_t x, const size_t y, const size_t w, const size_t h)
{
    T a = iImage.get(x, y);
    T b = iImage.get(x + w, y);
    T c = iImage.get(x, y + h);
    T d = iImage.get(x + w, y + h);

    return d - b - c + a;
}

template <typename T>
__global__ void calculate_iImage(T* iImage,const size_t width, const size_t height)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < height)
    {
        for (size_t i = 1; i < width; i++)
        {
            //iImage[j * width + i] = iImage[j * width + i] + iImage[j * width + i-1];
            atomicAdd(&(iImage[j * width + i]), iImage[j * width + i - 1]);
        }
    }

    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width)
    {
        for (size_t j = 1; j < height; j++)
        {
            atomicAdd(&(iImage[j * width + i]), iImage[(j - 1) * width + i]);
            //iImage[j * width + i] = iImage[j * width + i] + iImage[(j-1) * width + i];
        }
    }
}

int main()
{
    auto tStart = chrono::high_resolution_clock::now();
    Image<int> iImage_cpu;
    iImage_cpu.create(image_width, image_height);

    // allocate memory on cpu //
    cudaMallocHost((void**)&iImage_cpu.data, iImage_cpu.pixels * sizeof(int));
    createRandomImage(iImage_cpu);

    // allocate memory on gpu //
    Image<int> iImage_gpu;
    iImage_gpu.create(image_width, image_height);
    std::copy(&iImage_cpu.data[0], &iImage_cpu.data[0] + iImage_cpu.pixels, &iImage_gpu.data[0]);

    cudaMalloc((void**) & iImage_gpu.data, iImage_gpu.pixels * sizeof(int));

    // copy data to gpu //
    cudaMemcpy(iImage_gpu.data, iImage_cpu.data, iImage_gpu.pixels * sizeof(int), cudaMemcpyHostToDevice);

    // run calculation //
    calculate_iImage <<< gridSize, blockSize >>>(iImage_gpu.data, iImage_gpu.width, iImage_gpu.height);

    cudaMemcpy(iImage_cpu.data, iImage_gpu.data, iImage_gpu.pixels * sizeof(int), cudaMemcpyDeviceToHost);

    auto tEnd = chrono::high_resolution_clock::now();
    auto runtime = chrono::duration_cast<chrono::nanoseconds>(tEnd - tStart);

    printf("Time measured: %.3f seconds.\n", runtime.count() * 1e-9);

    //cout << iImage_cpu.get(999,999) << endl;
    // free gpu memory //
    cudaFree(iImage_gpu.data);

    // free cpu memory //
    cudaFreeHost(iImage_cpu.data);

    return 0;
}
