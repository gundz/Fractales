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

__global__ void
julia_kernel(t_cuda cuda, t_fractal fractal)
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

	pr = 0.003 * fractal.mx;
	pi = 0.003 * fractal.my;
	zx = fractal.cx + (x - cuda.rx / 2) * fractal.zoom + fractal.movex;
	zy = fractal.cy + (y - cuda.ry / 2) * fractal.zoom + fractal.movey;
	i = 0;
	while (++i < fractal.maxiteration)
	{
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = 2 * zx * zy + pi;
		zx = zx2 - zy2 + pr;
		if (zx2 + zy2 > 4)
			break ;
	}
	if (i == fractal.maxiteration)
		cuda.screen[dim_i] = 0xFFFFFFFF;
	else
		cuda.screen[dim_i] = (int)(i * 255 / fractal.maxiteration) << 24 | (i % 255)  << 16 | 255 << 8 | 255;
}

int
julia_call(t_data *data, t_cuda *cuda)
{
	static t_fractal fractal = {(2.5 / SDL_RY), 0, 0, 400, 0, 0, 0, 0, 0, 0};

	fractal.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	fractal.my = data->esdl->en.in.m_y - SDL_RY / 2;
	fractal_input(data, &fractal);
	julia_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, fractal);
	return (0);
}

void
julia(t_data *data)
{
	do_cuda(data, &julia_call);
}
