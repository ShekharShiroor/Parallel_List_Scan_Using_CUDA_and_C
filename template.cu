#include <wb.h>

#define MASK_WIDTH 5
#define O_TILE_WIDTH 16
#define clamp(x) (min(max((x), 0.0), 1.0))
#define BLOCK_WIDTH (O_TILE_WIDTH + MASK_WIDTH - 1)
#define mask_radius (MASK_WIDTH / 2)

//@@ INSERT CODE HERE 
//implement the tiled 2D convolution kernel with adjustments for channels
//use shared memory to reduce the number of global accesses, handle the boundary conditions when loading input list elements into the shared memory
//clamp your output values
__global__ void convolution_2D_kernel(float *P, const float* __restrict__ N, int height, int width, int channels, const float* __restrict__ M)
{
	__shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row_o = blockIdx.y*O_TILE_WIDTH + ty;
	int col_o = blockIdx.x*O_TILE_WIDTH + tx;
	int row_i = row_o - mask_radius;
	int col_i = col_o - mask_radius;

		if ((row_i >= 0) && (row_i < height) && (col_i >= 0) && (col_i < width))
		{
			Ns[ty][tx] = N[(row_i * width + col_i) * channels + blockIdx.z];
		}
		else {
			Ns[ty][tx] = 0.0f;
		}

		__syncthreads();

		float output = 0.0f;
		if (ty < O_TILE_WIDTH && tx < O_TILE_WIDTH)
		{
			for (int i = 0; i < MASK_WIDTH; i++) {
				for (int j = 0; j < MASK_WIDTH; j++) {
					output += M[i * MASK_WIDTH + j] * Ns[i + ty][j + tx];
				}
			}


			if (row_o < height && col_o < width)
				P[(row_o*width + col_o) * channels + blockIdx.z] = clamp(output);
		}

}

int main(int argc, char *argv[]) {
	wbArg_t arg;
	int maskRows;
	int maskColumns;
	int imageChannels;
	int imageWidth;
	int imageHeight;
	char *inputImageFile;
	char *inputMaskFile;
	wbImage_t inputImage;
	wbImage_t outputImage;
	float *hostInputImageData;
	float *hostOutputImageData;
	float *hostMaskData;
	float *deviceInputImageData;
	float *deviceOutputImageData;
	float *deviceMaskData;

	arg = wbArg_read(argc, argv); /* parse the input arguments */

	inputImageFile = wbArg_getInputFile(arg, 0);
	inputMaskFile = wbArg_getInputFile(arg, 1);

	inputImage = wbImport(inputImageFile);
	hostMaskData = (float *)wbImport(inputMaskFile, &maskRows, &maskColumns);

	assert(maskRows == MASK_WIDTH);    /* mask height is fixed to 5 */
	assert(maskColumns == MASK_WIDTH); /* mask width is fixed to 5 */

	imageWidth = wbImage_getWidth(inputImage);
	imageHeight = wbImage_getHeight(inputImage);
	imageChannels = wbImage_getChannels(inputImage);

	outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

	hostInputImageData = wbImage_getData(inputImage);
	hostOutputImageData = wbImage_getData(outputImage);

	wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

	wbTime_start(GPU, "Doing GPU memory allocation");
	//@@ INSERT CODE HERE
	//allocate device memory
	int sizeMask = sizeof(float) * maskColumns * maskRows;
	int sizeInputImage = sizeof(float) * imageWidth * imageHeight * imageChannels;

	cudaMalloc((void**)&deviceMaskData, sizeMask);
	cudaMalloc((void**)&deviceInputImageData, sizeInputImage);
	cudaMalloc((void**)&deviceOutputImageData, sizeInputImage);



	wbTime_stop(GPU, "Doing GPU memory allocation");

	wbTime_start(Copy, "Copying data to the GPU");
	//@@ INSERT CODE HERE
	//copy host memory to device
	cudaMemcpy(deviceMaskData, hostMaskData, sizeMask, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceInputImageData, hostInputImageData, sizeInputImage, cudaMemcpyHostToDevice);


	wbTime_stop(Copy, "Copying data to the GPU");

	wbTime_start(Compute, "Doing the computation on the GPU");
	//@@ INSERT CODE HERE
	//initialize thread block and kernel grid dimensions
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
	dim3 dimGrid(((imageWidth - 1) / O_TILE_WIDTH) + 1, ((imageHeight - 1) / O_TILE_WIDTH) + 1, imageChannels);

	//invoke CUDA kernel	
	convolution_2D_kernel << < dimGrid, dimBlock >> > (deviceOutputImageData, deviceInputImageData, imageHeight, imageWidth, imageChannels, deviceMaskData);

	wbTime_stop(Compute, "Doing the computation on the GPU");

	wbTime_start(Copy, "Copying data from the GPU");
	//@@ INSERT CODE HERE
	//copy results from device to host	
	cudaMemcpy(hostOutputImageData, deviceOutputImageData, sizeInputImage, cudaMemcpyDeviceToHost);


	wbTime_stop(Copy, "Copying data from the GPU");

	wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

	wbSolution(arg, outputImage);

	//@@ INSERT CODE HERE
	//deallocate device memory	
	cudaFree(deviceMaskData);
	cudaFree(deviceInputImageData);
	cudaFree(deviceOutputImageData);

	free(hostMaskData);
	wbImage_delete(outputImage);
	wbImage_delete(inputImage);

	return 0;
}
