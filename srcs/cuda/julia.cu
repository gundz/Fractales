#include <stdio.h>

extern "C" {
#include <header.h>
}

__global__ void
julia_kernel(Uint32 *a, int rx, int ry, int mx, int my)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * rx + x;
	if ((x >= rx) || (y >= ry))
		return ;

	//each iteration, it calculates: new = old*old + c, where c is a constant and old starts at current pixel
	double cRe, cIm;           //real and imaginary part of the constant c, determinate shape of the Julia Set
	double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old
	double zoom = 1, moveX = 0, moveY = 0; //you can change these to zoom and change position
	int maxIterations = 300; //after how much iterations the function should stop

	//pick some values for the constant c, this determines the shape of the Julia Set
	cRe = 0.001 * mx;
	cIm = 0.001 * my;
	//calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
	newRe = 1.5 * (x - rx / 2) / (0.5 * zoom * rx) + moveX;
	newIm = (y - ry / 2) / (0.5 * zoom * ry) + moveY;
	//i will represent the number of iterations
	int i;
	//start the iteration process
	for(i = 0; i < maxIterations; i++)
	{
		//remember value of previous iteration
		oldRe = newRe;
		oldIm = newIm;
		//the actual iteration, the real and imaginary part are calculated
		newRe = oldRe * oldRe - oldIm * oldIm + cRe;
		newIm = 2 * oldRe * oldIm + cIm;
		//if the point is outside the circle with radius 2: stop
		if ((newRe * newRe + newIm * newIm) > 4)
			break;
	}
	a[dim_i] = 0;
	if (i < maxIterations)
	{
		int color = 0;
		color = (color << 8) + i & 255;
		color = (color << 8) + 0;
		color = (color << 8) + 0;
		color = (color << 8) + 0xFF;
		a[dim_i] = color;
	}
	else if (i % 2 == 0)
		a[dim_i] = 0xFFFFFFFF;
}

extern "C" void
julia(t_data *data)
{
	static Uint32 *a_d = NULL;  // Pointer to host & device arrays

	size_t size = SDL_RY * SDL_RX * data->surf->format->BytesPerPixel;

	if (a_d == NULL)
		cudaMalloc((void **)&a_d, size);   // Allocate array on device

	dim3 blockSize(32, 32);
	int bx = (SDL_RX + blockSize.x - 1) / blockSize.x;
	int by = (SDL_RY + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(bx, by);

	julia_kernel<<<gridSize, blockSize>>>(a_d, SDL_RX, SDL_RY, data->esdl->en.in.m_x, data->esdl->en.in.m_y);
	cudaThreadSynchronize();

	SDL_LockSurface(data->surf);
	// Retrieve result from device and store it in host array
	cudaMemcpy(data->surf->pixels, a_d, size, cudaMemcpyDeviceToHost);
	SDL_UnlockSurface(data->surf);

	//cleanup
	if (data->esdl->run == 0)
		cudaFree(a_d);
}