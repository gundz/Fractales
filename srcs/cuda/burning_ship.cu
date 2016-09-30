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

__global__ void
burning_kernel(t_cuda cuda, t_fractal fractal)
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
		zx2 = zx * zx;
		zy2 = zy * zy;
		zy = fabs(2 * zx * zy) + pi;
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
burning_call(t_data *data, t_cuda *cuda)
{
	static t_fractal fractal = {(2.5 / 480), 0, 0, 400, 0, 0, 0, 0, 0, 0};

	fractal.mx = data->esdl->en.in.m_x - SDL_RX / 2;
	fractal.my = data->esdl->en.in.m_y - SDL_RY / 2;
	fractal_input(data, &fractal);
	burning_kernel<<<cuda->gridsize, cuda->blocksize>>>(*cuda, fractal);
	return (0);
}

void
burning_ship(t_data *data)
{
	do_cuda(data, &burning_call);
}
