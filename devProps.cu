#include<stdio.h>

int main()
{
int devcount;
cudaGetDeviceCount(&devcount);
printf("Device count:%d\n",devcount);
for (int i = 0; i < devcount; ++i){
// Get device properties
printf("\nCUDA Device #%d\n", i);
cudaDeviceProp devProp;
cudaGetDeviceProperties(&devProp, i);
printf("Name:%s\n", devProp.name);
printf("Compute capability: %d.%d\n",devProp.major ,devProp.minor);
printf("Warp Size %d\n",devProp.warpSize);
printf("Total global memory:%u bytes\n",devProp.totalGlobalMem);
printf("Total shared memory per block: %u bytes\n", devProp.sharedMemPerBlock);
printf("Total registers per block : %d\n",devProp.regsPerBlock);
printf("Clock rate: %d khz\n",devProp.clockRate);
printf("Maximum threads per block:%d\n", devProp.maxThreadsPerBlock);
for (int i = 0; i < 3; ++i)
printf("Maximum dimension %d of block: %d\n", i, devProp.maxThreadsDim[i]);
for (int i = 0; i < 3; ++i)
printf("Maximum dimension %d of grid: %d\n", i, devProp.maxGridSize[i]);
printf("Number of multiprocessors:%d\n", devProp.multiProcessorCount);
}
return 0;
}
