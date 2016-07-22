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

	float GraphTop = 1.5f;
	float GraphBottom = -1.5f;
	float GraphLeft = -2.0f;
	float GraphRight = 1.5f;
	int i;
	int max_iteration = 256;

	float incrementX = ((GraphRight - GraphLeft) / (rx - 1));
	float DecrementY = ((GraphTop - GraphBottom) / (ry - 1));
	float Zx, Zy;
	float CoordReal;
	float CoordImaginary = GraphTop;
	float SquaredX, SquaredY;

	int palette[256];
	for (int n = 0; n < 256; n++)
	{
		palette[n] = (int)(n + 512 - 512 * expf(-n / 50.0) / 3.0);
		palette[n] = palette[n] << 24 | palette[n] << 16 | palette[n] << 8 | 255;
	}
	palette[255] = 0;

	CoordReal = GraphLeft + (incrementX * x);
	CoordImaginary = GraphTop - (DecrementY * y);
	i = 0;
	Zx = CoordReal;
	Zy = CoordImaginary;
	SquaredX = Zx * Zx;
	SquaredY = Zy * Zy;
	a[dim_i] = 0;
	while ((i < max_iteration) && ((SquaredX + SquaredY) < 4.0))
	{
		Zy = fabs(Zx * Zy);
		Zy = Zy + Zy - CoordImaginary;
		Zx = SquaredX - SquaredY + CoordReal;
		SquaredX = Zx * Zx;
		SquaredY = Zy * Zy;
		i++;
	}
	a[dim_i] = palette[(i + 1) % 255];
}

void
set_palette(int palette[256])
{
	for (int n = 0; n < 256; n++)
	{
		palette[n] = (int)(n + 512 - 512 * expf(-n / 50.0) / 3.0);
		palette[n] = palette[n] << 24 | palette[n] << 16 | palette[n] << 8 | 255;
	}
	palette[255] = 0;
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

	//int palette[256];
	//set_palette(palette);
	burning_ship_kernel<<<gridSize, blockSize>>>(a_d, SDL_RX, SDL_RY);
	cudaThreadSynchronize();

	SDL_LockSurface(data->surf);
	// Retrieve result from device and store it in host array
	cudaMemcpy(data->surf->pixels, a_d, size, cudaMemcpyDeviceToHost);
	SDL_UnlockSurface(data->surf);

	//cleanup
	if (data->esdl->run == 0)
		cudaFree(a_d);
}