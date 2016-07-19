#include <stdio.h>

extern "C" {
#include <header.h>
}

__global__ void
burning_ship_kernel(Uint32 *a, int rx, int ry)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * rx + x;
	if ((x >= rx) || (y >= ry))
		return ;
/*
	double x1 = -2.1;
	double x2 = 0.6;
	double y1 = -1.2;
	double y2 = 1.2;
	double zoom = 100;
	int maxIteration = 100;
	const int img_x = (x2 - x2) * zoom;
	const int img_y = (y2 - y1) * zoom;

	int pixels[img_y][img_x];

	double c_r = y / zoom + x1;
	double c_i = y / zoom + y1;
	double z_r = 0;
	double z_i = 0;
	int i = 0;
	while (z_r * z_r + z_i * z_i < 4 && i < maxIteration)
	{
		int tmp = z_r;
		z_r = z_r * z_r - z_i * z_i + c_r;
		z_i = 2 * z_i * tmp + c_i;
		i++;

	}
*/

	a[dim_i] = 0;
}

extern "C" void
burning_ship(t_data *data)
{
	static Uint32 *a_d = NULL;  // Pointer to host & device arrays

	size_t size = SDL_RY * SDL_RX * data->surf->format->BytesPerPixel;

	if (a_d == NULL)
		cudaMalloc((void **)&a_d, size);   // Allocate array on device

	dim3 blockSize(32, 32);
	int bx = (SDL_RX + blockSize.x - 1) / blockSize.x;
	int by = (SDL_RY + blockSize.y - 1) / blockSize.y;
	dim3 gridSize = dim3(bx, by);

	buddhabrot_kernel<<<gridSize, blockSize>>>(a_d, SDL_RX, SDL_RY);
	cudaThreadSynchronize();

	SDL_LockSurface(data->surf);
	// Retrieve result from device and store it in host array
	cudaMemcpy(data->surf->pixels, a_d, size, cudaMemcpyDeviceToHost);
	SDL_UnlockSurface(data->surf);

	//cleanup
	if (data->esdl->run == 0)
		cudaFree(a_d);
}