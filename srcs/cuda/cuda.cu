extern "C"
{
#include <header.h>
}
#include <cuda.h>

//SOME CHECK TO DO MALLOC ETC...

void
do_cuda(t_data *data, int (*f)(t_data *, t_cuda *))
{
	static t_cuda cuda = {NULL};

	size_t size = SDL_RY * SDL_RX * data->surf->format->BytesPerPixel;

	if (cuda.screen == NULL)
	{
		cudaMalloc((void **)&cuda.screen, size);
		cuda.blockSize = dim3(32, 32);
		cuda.bx = (SDL_RX + cuda.blockSize.x - 1) / cuda.blockSize.x;
		cuda.by = (SDL_RY + cuda.blockSize.y - 1) / cuda.blockSize.y;
		cuda.gridSize = dim3(cuda.bx, cuda.by);
		cuda.rx = SDL_RX;
		cuda.ry = SDL_RY;
	}

	f(data, &cuda);
	cudaDeviceSynchronize();

	SDL_LockSurface(data->surf);
	cudaMemcpy(data->surf->pixels, cuda.screen, size, cudaMemcpyDeviceToHost);
	SDL_UnlockSurface(data->surf);

	if (data->esdl->run == 0)
		cudaFree(cuda.screen);
}