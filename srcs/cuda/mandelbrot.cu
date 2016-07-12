#include <stdio.h>

extern "C" {
#include <header.h>
}

__global__ void
my_kernel(int *a, int rx, int ry)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * rx + x;
	if ((x >= rx) || (y >= ry))
		return ;

	//each iteration, it calculates: newz = oldz*oldz + p, where p is the current pixel, and oldz stars at the origin
	double pr, pi;           //real and imaginary part of the pixel p
	double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old z
	double zoom = 1, moveX = -0.5, moveY = 0; //you can change these to zoom and change position
	int maxIterations = 300;//after how much iterations the function should stop

	//calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
	pr = 1.5 * (x - rx / 2) / (0.5 * zoom * rx) + moveX;
	pi = (y - ry / 2) / (0.5 * zoom * ry) + moveY;
	newRe = newIm = oldRe = oldIm = 0; //these should start at 0,0

	int i;
	//start the iteration process
	for(i = 0; i < maxIterations; i++)
	{
		//remember value of previous iteration
		oldRe = newRe;
		oldIm = newIm;
		//the actual iteration, the real and imaginary part are calculated
		newRe = oldRe * oldRe - oldIm * oldIm + pr;
		newIm = 2 * oldRe * oldIm + pi;
		//if the point is outside the circle with radius 2: stop
		if ((newRe * newRe + newIm * newIm) > 4)
			break;
	}
	//use color model conversion to get rainbow palette, make brightness black if maxIterations reached
	//color = HSVtoRGB(ColorHSV(i % 256, 255, 255 * (i < maxIterations)));
	//draw the pixel
	if (i < maxIterations)
		a[dim_i] = 0xFFFFFFFF;
}

extern "C" void
mandelbrot(t_data *data, SDL_Surface *surf)
{
	static int *a_h = NULL;
	static int *a_d = NULL;  // Pointer to host & device arrays

	size_t size = SDL_RX * SDL_RY * sizeof(int);

	if (a_h == NULL)
		a_h = (int *)malloc(size);        // Allocate array on host

	if (a_d == NULL)
		cudaMalloc((void **) &a_d, size);   // Allocate array on device

	// Initialize host array and copy it to CUDA device
	for (int i = 0; i < SDL_RX * SDL_RY; i++)
		a_h[i] = (int)0;
	
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	dim3 blockSize(32, 32);
	int bx = (SDL_RX + blockSize.x - 1) / blockSize.x;
	int by = (SDL_RY + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(bx, by);

	my_kernel <<< gridSize, blockSize >>> (a_d, SDL_RX, SDL_RY);

	//cudaThreadSynchronize();

	// Retrieve result from device and store it in host array
	cudaMemcpy(a_h, a_d, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < SDL_RY; i++)
	{
		for (int j = 0; j < SDL_RX; j++)
		{
			Esdl_put_pixel(surf, j, i, a_h[i * SDL_RX + j]);
		}
	}

	//cleanup
	if (data->esdl->run == 0)
	{
		printf("QUIT BITCH !\n");
		free(a_h);
		cudaFree(a_d);
	}

	//my_kernel<<<1,1>>>();
	//cudaDeviceSynchronize();



	(void)data;
	(void)surf;
}