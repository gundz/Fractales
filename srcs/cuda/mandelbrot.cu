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
#include <stdlib.h>

__device__ int
mandelbrot_color(double new_re, double new_im, int i, int max_iteration)
{
	double		z;
	int			brightness;

	z = sqrt(new_re * new_re + new_im * new_im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness << 24 | (i * 255) / (max_iteration % 255) << 16 | i % 255 << 8 | 255);
}

__global__ void
mandelbrot_kernel(t_cuda cuda, t_mandelbrot mandelbrot)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	const double c_r = x / mandelbrot.zoomx + mandelbrot.x1;
	const double c_i = y / mandelbrot.zoomy + mandelbrot.y1;
	double z_r = 0;
	double z_i = 0;
	double tmp;
	int			i;

	i = 0;
	while (((z_r * z_r + z_i * z_i) < 4) && i < mandelbrot.maxiteration)
	{
	    tmp = z_r;
	    z_r = z_r * z_r - z_i * z_i + c_r;
	    z_i = 2 * z_i * tmp + c_i;
	    i++;
	}
	if (i == mandelbrot.maxiteration)
		cuda.screen[dim_i] = 0xFFFFFFFF;
	else
		cuda.screen[dim_i] = mandelbrot_color(z_r, z_i, i, mandelbrot.maxiteration);
}

t_mandelbrot
mandelbrot_zoomin(t_data *data, t_mandelbrot mandelbrot)
{
	t_mandelbrot new_mandelbrot;

	new_mandelbrot.zoomx = mandelbrot.zoomx * 2;
	new_mandelbrot.zoomy = mandelbrot.zoomy * 2;

	double X = mandelbrot.x1 + data->esdl->en.in.m_x * (mandelbrot.x2 - mandelbrot.x1) / SDL_RX;
	double Y = mandelbrot.y1 + data->esdl->en.in.m_y * (mandelbrot.y2 - mandelbrot.y1) / SDL_RY;

	new_mandelbrot.x1 = X - (mandelbrot.x2 - mandelbrot.x1) / 4;
	new_mandelbrot.x2 = X + (mandelbrot.x2 - mandelbrot.x1) / 4;
	new_mandelbrot.y1 = Y - (mandelbrot.y2 - mandelbrot.y1) / 4;
	new_mandelbrot.y2 = Y + (mandelbrot.y2 - mandelbrot.y1) / 4;

	if (mandelbrot.maxiteration < 8000)
		new_mandelbrot.maxiteration = mandelbrot.maxiteration * 1.1;
	else
		new_mandelbrot.maxiteration = mandelbrot.maxiteration;

	return (new_mandelbrot);
}

t_mandelbrot
mandelbrot_zoomout(t_data *data, t_mandelbrot mandelbrot)
{
	t_mandelbrot new_mandelbrot;

	new_mandelbrot.zoomx = mandelbrot.zoomx / 2;
	new_mandelbrot.zoomy = mandelbrot.zoomy / 2;

	double X = mandelbrot.x1 + data->esdl->en.in.m_x * (mandelbrot.x2 - mandelbrot.x1) / SDL_RX;
	double Y = mandelbrot.y1 + data->esdl->en.in.m_y * (mandelbrot.y2 - mandelbrot.y1) / SDL_RY;

	new_mandelbrot.x1 = X - (mandelbrot.x2 - mandelbrot.x1);
	new_mandelbrot.x2 = X + (mandelbrot.x2 - mandelbrot.x1);
	new_mandelbrot.y1 = Y - (mandelbrot.y2 - mandelbrot.y1);
	new_mandelbrot.y2 = Y + (mandelbrot.y2 - mandelbrot.y1);

	if (mandelbrot.maxiteration > 200)
		new_mandelbrot.maxiteration = mandelbrot.maxiteration * 0.9;
	else
		new_mandelbrot.maxiteration = mandelbrot.maxiteration;
	return (new_mandelbrot);
}

int
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	static t_mandelbrot	mandelbrot = {
		-2.1, 0.6, -1.2, 1.2,
		SDL_RX / (0.6 - (-2.1)),
		SDL_RY / (1.2 - (-1.2)),
		200
		};

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		mandelbrot = mandelbrot_zoomin(data, mandelbrot);
		data->esdl->en.in.button[SDL_BUTTON_LEFT] = 0;
	}

	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		mandelbrot = mandelbrot_zoomout(data, mandelbrot);
		data->esdl->en.in.button[SDL_BUTTON_RIGHT] = 0;
	}

	mandelbrot_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, mandelbrot);
	return (0);
}

void
mandelbrot(t_data *data)
{
	do_cuda(data, &mandelbrot_call);
}
