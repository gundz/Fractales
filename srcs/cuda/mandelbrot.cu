/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mandelbrot.cu                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:28:52 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:28:54 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <mandelbrot.h>

__device__ int
mandelbrot_color(double new_re, double new_im, int i, int max_iteration)
{
	double		z;
	int			brightness;

	//z = sqrt(new_re * new_re + new_im * new_im);
	(void)z;
	brightness = 256. * log2(1.75 + i - log2(log2((double)(max_iteration / 3)))) / log2((double)(max_iteration));
	return (brightness << 24 | (i % 255)  << 16 | brightness << 8 | 255);
}

__global__ void
mandelbrot_kernel(t_cuda cuda, t_mandelbrot mandelbrot)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	double		pr;
	double		pi;
	double		zx;
	double		zy;
	double		zx2;
	double		zy2;
	int			i;

	pr = (x - cuda.rx / 2) / (mandelbrot.zoom * cuda.rx) + mandelbrot.movex;
	pi = (y - cuda.ry / 2) / (mandelbrot.zoom * cuda.ry) + mandelbrot.movey;

	zx = zy = zx2 = zy2 = 0;
	i = 0;
	while ((zx2 + zy2 < 4) && i < mandelbrot.maxiteration)
	{
		zy = 2 * zx * zy + pi;
		zx = zx2 - zy2 + pr;
		zx2 = zx * zx;
		zy2 = zy * zy;
		i++;
	}
	if (i == mandelbrot.maxiteration)
		cuda.screen[dim_i] = 0xFFFFFFFF;
	else
		cuda.screen[dim_i] = i * 255 / mandelbrot.caca << 24 | (i % 255)  << 16 | 127 << 8 | 255;
}

void
mandelbrot_input(t_data *data, t_mandelbrot *mandelbrot)
{
	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		mandelbrot->movex -= 0.01 / mandelbrot->zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		mandelbrot->movex += 0.01 / mandelbrot->zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		mandelbrot->movey -= 0.01 / mandelbrot->zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		mandelbrot->movey += 0.01 / mandelbrot->zoom * 10;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		mandelbrot->zoom += 0.05 * mandelbrot->zoom;
		mandelbrot->maxiteration *= 1.0050;
		printf("Max Iteration = %d\n", (int)mandelbrot->maxiteration);
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		mandelbrot->zoom -= 0.05 * mandelbrot->zoom;
		mandelbrot->maxiteration *= 0.9950;
		printf("Max Iteration = %d\n", (int)mandelbrot->maxiteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		mandelbrot->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		mandelbrot->maxiteration *= 0.9;
}

int
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	static t_mandelbrot	mandelbrot = {1, 0, 0, 200, 300};

	mandelbrot_input(data, &mandelbrot);
	mandelbrot_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, mandelbrot);
	return (0);
}

void
mandelbrot(t_data *data)
{
	do_cuda(data, &mandelbrot_call);
}
