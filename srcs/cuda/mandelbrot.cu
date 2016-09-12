extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <mandelbrot.h>

__global__ void
mandelbrot_kernel(t_cuda cuda, t_mandelbrot mandelbrot)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	//each iteration, it calculates: newz = oldz*oldz + p, where p is the current pixel, and oldz stars at the origin
	double cr, ci;           //real and imaginary part of the pixel p
	double zr, zi;
	double temp;
	//calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
	cr = 1.5 * (x - cuda.rx / 2) / (0.5 * mandelbrot.zoom * cuda.rx) + mandelbrot.moveX;
	ci = (y - cuda.ry / 2) / (0.5 * mandelbrot.zoom * cuda.ry) + mandelbrot.moveY;
	zr = zi = 0;

	int i = 0;
	while ((zr * zr + zi * zi < 4) && i < mandelbrot.maxIteration)
	{
		temp = zr * zr - zi * zi + cr;
		zi = (2 * zr * zi + ci);
		zr = temp;
		i++;
	}
	cuda.screen[dim_i] = (i % 256) << 24 | (i % 256) << 16 | (i % 256) << 8 | 255;
}

int
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	static t_mandelbrot	mandelbrot = {1, -0.5, 0, 300, {0}};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		mandelbrot.moveX -= 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		mandelbrot.moveX += 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		mandelbrot.moveY -= 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		mandelbrot.moveY += 0.01 / mandelbrot.zoom * 10;

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		mandelbrot.zoom += 0.01 * mandelbrot.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		mandelbrot.zoom -= 0.01 * mandelbrot.zoom;

	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		mandelbrot.maxIteration *= 1.1;
		printf("Max iterations = %d\n", mandelbrot.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1 && mandelbrot.maxIteration > 300)
	{
		mandelbrot.maxIteration /= 1.1;
		printf("Max iterations = %d\n", mandelbrot.maxIteration);
	}


	mandelbrot_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, mandelbrot);
	return (0);
}

void
mandelbrot(t_data *data)
{
	do_cuda(data, &mandelbrot_call);
}