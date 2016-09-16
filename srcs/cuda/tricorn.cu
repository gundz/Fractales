/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tricorn.cu                                         :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/16 15:30:16 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/16 15:30:17 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <tricorn.h>

__global__ void
tricorn_kernel(t_cuda cuda, t_tricorn tricorn)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

    double pr, pi;
    double newRe, newIm, oldRe, oldIm;

	pr = (x - cuda.rx / 2) / (0.5 * tricorn.zoom * cuda.rx) + tricorn.moveX;
	pi = (y - cuda.ry / 2) / (0.5 * tricorn.zoom * cuda.ry) + tricorn.moveY;
	newRe = newIm = oldRe = oldIm = 0;

	int i = 0;
	while (((newRe * newRe + newIm * newIm) < 4) && i < tricorn.maxIteration)
	{
	    oldRe = newRe;
	    oldIm = newIm;
	    newRe = oldRe * oldRe - oldIm * oldIm + pr;
	    newIm = -(2 * oldRe * oldIm + pi);
	    i++;
	}

    if(i == tricorn.maxIteration)
        cuda.screen[dim_i] = 0x00000000;
    else
    {
        double z = sqrt(newRe * newRe + newIm * newIm);
        int brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2(double(tricorn.maxIteration));
        cuda.screen[dim_i] = brightness << 24 | (i % 255) << 16 | 255 << 8 | 255;
    }
}

int
tricorn_call(t_data *data, t_cuda *cuda)
{
	static t_tricorn	tricorn = {1, -0.5, 0, 200};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		tricorn.moveX -= 0.01 / tricorn.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		tricorn.moveX += 0.01 / tricorn.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		tricorn.moveY -= 0.01 / tricorn.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		tricorn.moveY += 0.01 / tricorn.zoom * 10;

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		tricorn.zoom += 0.01 * tricorn.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		tricorn.zoom -= 0.01 * tricorn.zoom;

	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		tricorn.maxIteration *= 1.1;
		printf("Max iterations = %d\n", tricorn.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
	{
		tricorn.maxIteration *= 0.9;
		printf("Max iterations = %d\n", tricorn.maxIteration);
	}


	tricorn_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, tricorn);
	return (0);
}

void
tricorn(t_data *data)
{
	do_cuda(data, &tricorn_call);
}
