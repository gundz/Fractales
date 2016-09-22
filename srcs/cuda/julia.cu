/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   julia.cu                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:28:59 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:29:00 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

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

	double		pr, pi;
	double		zx, zy;
	double		zx2, zy2;
	int			i;

	pr = 0.003 * julia.mx;
	pi = 0.003 * julia.my;
	zx = julia.cx + (x - cuda.rx / 2) * julia.zoom + julia.movex;
	zy = julia.cy + (y - cuda.ry / 2) * julia.zoom + julia.movey;
	i = 0;
	while (i < julia.maxiteration)
	{
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = 2 * zx * zy + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 > 4)
			break ;
		i++;
	}
	if (i == julia.maxiteration)
		cuda.screen[dim_i] = 0xFFFFFFFF;
	else
		cuda.screen[dim_i] = (int)(i * 255 / julia.maxiteration) << 24 | (i % 255)  << 16 | 255 << 8 | 255;
}

void
julia_input(t_data *data, t_julia *julia)
{
	julia->oldcx = julia->cx;
	julia->oldcy = julia->cy;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		julia->movex -= 0.0001 / julia->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		julia->movex += 0.0001 / julia->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		julia->movey -= 0.0001 / julia->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		julia->movey += 0.0001 / julia->zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		julia->zoom = julia->zoom / 1.05;
		julia->cx = (julia->oldcx) + (julia->mx * 0.05) * julia->zoom;
		julia->cy = (julia->oldcy) + (julia->my * 0.05) * julia->zoom;
		julia->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		julia->zoom = julia->zoom * 1.05;
		julia->cx = (julia->oldcx) + (julia->mx * 0.05) * julia->zoom;
		julia->cy = (julia->oldcy) + (julia->my * 0.05) * julia->zoom;
		julia->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		julia->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		julia->maxiteration *= 0.9;
}

int
julia_call(t_data *data, t_cuda *cuda)
{
	static t_julia julia = {(2.5 / SDL_RY), 0, 0, 400, 0, 0, 0, 0, 0, 0};

	julia.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	julia.my = data->esdl->en.in.m_y - SDL_RY / 2;
	julia_input(data, &julia);
	julia_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, julia);
	return (0);
}

void
julia(t_data *data)
{
	do_cuda(data, &julia_call);
}
