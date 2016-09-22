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
burning_kernel(t_cuda cuda, t_burning burning)
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

	pr = burning.cx + (x - cuda.rx / 2) * burning.zoom + burning.movex;
	pi = burning.cy + (y - cuda.ry / 2) * burning.zoom + burning.movey;
	zx = 0;
	zy = 0;
	i = 0;
	while (i < burning.maxiteration)
	{
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = fabs(2 * zx * zy) + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 > 4)
			break ;
		i++;
	}
	if (i == burning.maxiteration)
		cuda.screen[dim_i] = 0xFFFFFFFF;
	else
		cuda.screen[dim_i] = (int)(i * 255 / burning.maxiteration) << 24 | (i % 255)  << 16 | 255 << 8 | 255;
}

void
burning_input(t_data *data, t_burning *burning)
{
	burning->oldcx = burning->cx;
	burning->oldcy = burning->cy;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		burning->movex -= 0.0001 / burning->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		burning->movex += 0.0001 / burning->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		burning->movey -= 0.0001 / burning->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		burning->movey += 0.0001 / burning->zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		burning->zoom = burning->zoom / 1.05;
		burning->cx = (burning->oldcx) + (burning->mx * 0.05) * burning->zoom;
		burning->cy = (burning->oldcy) + (burning->my * 0.05) * burning->zoom;
		burning->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		burning->zoom = burning->zoom * 1.05;
		burning->cx = (burning->oldcx) + (burning->mx * 0.05) * burning->zoom;
		burning->cy = (burning->oldcy) + (burning->my * 0.05) * burning->zoom;
		burning->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		burning->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		burning->maxiteration *= 0.9;
}

int
burning_call(t_data *data, t_cuda *cuda)
{
	static t_burning burning = {(2.5 / SDL_RY), 0, 0, 400, 0, 0, 0, 0, 0, 0};

	burning.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	burning.my = data->esdl->en.in.m_y - SDL_RY / 2;
	burning_input(data, &burning);
	burning_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, burning);
	return (0);
}

void
burning_ship(t_data *data)
{
	do_cuda(data, &burning_call);
}
