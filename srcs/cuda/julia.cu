extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <julia.h>

__global__ void
julia_kernel(t_cuda cuda, t_julia julia)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	//each iteration, it calculates: new = old*old + c, where c is a constant and old starts at current pixel
	double cRe, cIm;           //real and imaginary part of the constant c, determinate shape of the Julia Set
	double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old

	//pick some values for the constant c, this determines the shape of the Julia Set
	cRe = 0.001 * julia.mx;
	cIm = 0.001 * julia.my;
	//calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
	newRe = 1.5 * (x - cuda.rx / 2) / (0.5 * julia.zoom * cuda.rx) + julia.moveX;
	newIm = (y - cuda.ry / 2) / (0.5 * julia.zoom * cuda.ry) + julia.moveY;
	//i will represent the number of iterations
	int i;
	for(i = 0; i < julia.maxIteration; i++)
	{
		oldRe = newRe;
		oldIm = newIm;
		newRe = oldRe * oldRe - oldIm * oldIm + cRe;
		newIm = 2 * oldRe * oldIm + cIm;
		if ((newRe * newRe + newIm * newIm) > 4)
			break;
	}
	cuda.screen[dim_i] = julia.palette[(i + 1) % 255];
}

static void
set_palette(int palette[256])
{
	palette[0] = 0;
	for (int i = 1; i < 256; i++)
	{
		palette[i] = (int)(i + 512 - 512 * expf(-i / 50.0) / 3.0);
		palette[i] = palette[i] << 24 | i % 255 << 16 | palette[i] % 255 << 8 | 255;
	}
}

int
julia_call(t_data *data, t_cuda *cuda)
{
	static t_julia julia = {0, 0, 1, 0, 0, 300, {0}};
	julia.mx = data->esdl->en.in.m_x;
	julia.my = data->esdl->en.in.m_y;
	set_palette(julia.palette);

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
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1 && julia.maxIteration > 300)
	{
		julia.maxIteration /= 1.1;
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