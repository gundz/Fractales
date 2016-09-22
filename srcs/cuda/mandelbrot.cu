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

	double		pr, pi;
	double		zx, zy;
	double		zx2, zy2;
	int			i;

	pr = mandelbrot.cx + (x - cuda.rx / 2) * mandelbrot.zoom + mandelbrot.movex;
	pi = mandelbrot.cy + (y - cuda.ry / 2) * mandelbrot.zoom + mandelbrot.movey;
	zx = 0;
	zy = 0;
	i = 0;
	while (i < mandelbrot.maxiteration)
	{
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = 2 * zx * zy + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 >= 4)
			break ;
		i++;
	}
	if (i == mandelbrot.maxiteration)
		cuda.screen[dim_i] = 0xFFFFFFFF;
	else
		cuda.screen[dim_i] = (int)(i * 255 / mandelbrot.maxiteration) << 24 | (i % 255)  << 16 | 255 << 8 | 255;
}

/*mandelbrot
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = 2 * zx * zy + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 >= 4)
			break ;
*/

/*
mandelbrot2
		zy2 = zy * zy;
		zx2 = zx * zx;
		zx = (zx2 * zx) - 3 * zx * zy2 + pr;
		zy = 3 * zx2 * zy - (zy2 * zy) + pi;
		if (zx2 + zy2 >= 4)
			break ;
*/

/*
mandelbrot4
		zy2 = zy * zy;
		zx2 = zx * zx;
		zx = (zx2 * zx2) - (6 * zx2 * zy2) + (zy2 * zy2) + pr;
		zy = (4 * (zx2 * zx) * zy) - (4 * zx * (zy2 * zy)) + pi;
		if (zx2 + zy2 >= 4)
			break ;
*/

void
mandelbrot_input(t_data *data, t_mandelbrot *mandelbrot)
{
	mandelbrot->oldcx = mandelbrot->cx;
	mandelbrot->oldcy = mandelbrot->cy;

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		mandelbrot->movex -= 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		mandelbrot->movex += 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		mandelbrot->movey -= 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		mandelbrot->movey += 0.0001 / mandelbrot->zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		mandelbrot->zoom = mandelbrot->zoom / 1.05;
		mandelbrot->cx = (mandelbrot->oldcx) + (mandelbrot->mx * 0.05) * mandelbrot->zoom;
		mandelbrot->cy = (mandelbrot->oldcy) + (mandelbrot->my * 0.05) * mandelbrot->zoom;
		mandelbrot->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		mandelbrot->zoom = mandelbrot->zoom * 1.05;
		mandelbrot->cx = (mandelbrot->oldcx) + (mandelbrot->mx * 0.05) * mandelbrot->zoom;
		mandelbrot->cy = (mandelbrot->oldcy) + (mandelbrot->my * 0.05) * mandelbrot->zoom;
		mandelbrot->maxiteration *= 0.9975;
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		mandelbrot->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		mandelbrot->maxiteration *= 0.9;
}

int
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	static t_mandelbrot	mandelbrot = {(2.5 / SDL_RY), 0, 0, 400, 0, 0, 0, 0, 0, 0};

	mandelbrot.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	mandelbrot.my = data->esdl->en.in.m_y - SDL_RY / 2;
	mandelbrot_input(data, &mandelbrot);
	mandelbrot_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, mandelbrot);
	return (0);
}

void
mandelbrot(t_data *data)
{
	do_cuda(data, &mandelbrot_call);
}
