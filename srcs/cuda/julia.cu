# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    julia.cu                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2016/09/16 15:35:50 by fgundlac          #+#    #+#              #
#    Updated: 2016/09/16 15:35:51 by fgundlac         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <julia.h>

__device__ int
julia_color(double new_re, double new_im, int i, int max_iteration)
{
	double			z;
	int				brightness;

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | (i % 255) << 16 | 255 << 8 | 255);
}

__global__ void
julia_kernel(t_cuda cuda, t_julia julia)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

   	double			pr;
	double			pi;
	double			new_re;
	double			new_im;
	double			old_re;
	double			old_im;
	int				i;

	pr = 0.001 * julia.mx;
	pi = 0.001 * julia.my;
	new_re = (x - cuda.rx / 2) / (0.5 * julia.zoom * cuda.rx) + julia.moveX;
	new_im = (y - cuda.ry / 2) / (0.5 * julia.zoom * cuda.ry) + julia.moveY;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < julia.maxIteration)
	{
		old_re = new_re;
		old_im = new_im;
		new_re = old_re * old_re - old_im * old_im + pr;
		new_im = 2 * old_re * old_im + pi;
		i++;
	}
	cuda.screen[dim_i] = julia_color(new_re, new_im, i, julia.maxIteration);
}

int
julia_call(t_data *data, t_cuda *cuda)
{
	static t_julia julia = {0, 0, 1, 0, 0, 300};
	julia.mx = data->esdl->en.in.m_x;
	julia.my = data->esdl->en.in.m_y;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		julia.moveX -= 0.01 / julia.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		julia.moveX += 0.01 / julia.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		julia.moveY -= 0.01 / julia.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		julia.moveY += 0.01 / julia.zoom * 10;

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		julia.zoom += 0.01 * julia.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		julia.zoom -= 0.01 * julia.zoom;

	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		julia.maxIteration *= 1.1;
		printf("Max iterations = %d\n", julia.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
	{
		julia.maxIteration *= 0.9;
		printf("Max iterations = %d\n", julia.maxIteration);
	}

	julia_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, julia);
	return (0);
}

void
julia(t_data *data)
{
	do_cuda(data, &julia_call);
}
