extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include "tools.cu"

__global__ void
mandelbrot4_kernel(t_cuda cuda, t_fractal fractal)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	double		pr, pi;
	double		zx, zy;
	double		zx2, zy2;
	int			i;

	pr = fractal.cx + (x - cuda.rx / 2) * fractal.zoom + fractal.movex;
	pi = fractal.cy + (y - cuda.ry / 2) * fractal.zoom + fractal.movey;
	zx = 0;
	zy = 0;
	i = 0;
	while (++i < fractal.maxiteration)
	{
		zy2 = zy * zy;
		zx2 = zx * zx;
		zx = (zx2 * zx2) - (6 * zx2 * zy2) + (zy2 * zy2) + pr;
		zy = (4 * (zx2 * zx) * zy) - (4 * zx * (zy2 * zy)) + pi;
		if (zx2 + zy2 >= 4)
			break ;
	}
	int brightness = cuda_color_it(zx2, zy2, i, 100);
	cuda.screen[dim_i] = hsv_to_rgb(brightness % 256, 255, 255 * (i < fractal.maxiteration));
}

int
mandelbrot4_call(t_data *data, t_cuda *cuda)
{
	static t_fractal fractal = {(2.5 / SDL_RY), 0, 0, 400, 0, 0, 0, 0, 0, 0};

	fractal.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	fractal.my = data->esdl->en.in.m_y - SDL_RY / 2;
	fractal_input(data, &fractal);
	mandelbrot4_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, fractal);
	return (0);
}

void
mandelbrot4(t_data *data)
{
	do_cuda(data, &mandelbrot4_call);
}
