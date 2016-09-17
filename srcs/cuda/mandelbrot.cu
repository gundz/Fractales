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

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | (i % 255) << 16 | 255 << 8 | 255);
}

__global__ void
mandelbrot_kernel(t_cuda cuda, t_mandelbrot mandelbrot)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	double		pr;
	double		pi;
	double		new_re;
	double		new_im;
	double		old_re;
	double		old_im;
	int			i;

	pr = (x - cuda.rx / 2) / (0.5 * mandelbrot.zoom * cuda.rx) + mandelbrot.movex;
	pi = (y - cuda.ry / 2) / (0.5 * mandelbrot.zoom * cuda.ry) + mandelbrot.movey;
	new_re = new_im = old_re = old_im = 0;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < mandelbrot.maxiteration)
	{
		old_re = new_re;
		old_im = new_im;
		new_re = old_re * old_re - old_im * old_im + pr;
		new_im = 2 * old_re * old_im + pi;
		i++;
	}
	cuda.screen[dim_i] = mandelbrot_color(new_re, new_im, i, mandelbrot.maxiteration);
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
		mandelbrot->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		mandelbrot->zoom -= 0.05 * mandelbrot->zoom;
		mandelbrot->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		mandelbrot->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		mandelbrot->maxiteration *= 0.9;
	printf("Max Iterations: %f\n", mandelbrot->maxiteration);
}

int
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	static t_mandelbrot	mandelbrot = {1, -0.5, 0, 200};

	mandelbrot_input(data, &mandelbrot);
	mandelbrot_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, mandelbrot);
	return (0);
}

void
mandelbrot(t_data *data)
{
	do_cuda(data, &mandelbrot_call);
}
