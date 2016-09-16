/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   burning_ship.cu                                    :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:29:21 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:29:22 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <burning_ship.h>

__device__ int
burning_ship_color(double new_re, double new_im, int i, int max_iteration)
{
	double		z;
	int			brightness;

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | brightness << 16 | brightness << 8 | 255);
}

__global__ void
burning_ship_kernel(t_cuda cuda, t_burning burning)
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

	pr = (x - cuda.rx / 2) / (0.5 * burning.zoom * cuda.rx) + burning.moveX;
	pi = (y - cuda.ry / 2) / (0.5 * burning.zoom * cuda.ry) + burning.moveY;
	new_re = new_im = old_re = old_im = 0;
	i = 0;
	while (((new_re * new_re + new_im * new_im) < 4) && i < burning.maxIteration)
	{
		new_re = (old_re * old_re - old_im * old_im) + pr;
		new_im = (fabs(old_re * old_im) * 2) + pi;
		old_re = new_re;
		old_im = new_im;
		i++;
	}
	cuda.screen[dim_i] = burning_ship_color(new_re, new_im, i, burning.maxIteration);
}

int
burning_call(t_data *data, t_cuda *cuda)
{
	static t_burning	burning = {1, 0, 0, 200};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		burning.moveX -= 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		burning.moveX += 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		burning.moveY -= 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		burning.moveY += 0.01 / burning.zoom * 10;

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		burning.zoom += 0.05 * burning.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		burning.zoom -= 0.05 * burning.zoom;

	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		burning.maxIteration *= 1.1;
		printf("Max iterations = %d\n", burning.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
	{
		burning.maxIteration *= 0.9;
		printf("Max iterations = %d\n", burning.maxIteration);
	}


	burning_ship_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, burning);
	return (0);
}

void
burning_ship(t_data *data)
{
	do_cuda(data, &burning_call);
}
