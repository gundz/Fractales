extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <mandelbrot.h>

__global__ void
mandelbrot_kernel(t_cuda cuda, t_mandelbrot mandelbrot)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	//each iteration, it calculates: newz = oldz*oldz + p, where p is the current pixel, and oldz stars at the origin
	double pr, pi;           //real and imaginary part of the pixel p
	double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old z

	//calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
	pr = 1.5 * (x - cuda.rx / 2) / (0.5 * mandelbrot.zoom * cuda.rx) + mandelbrot.moveX;
	pi = (y - cuda.ry / 2) / (0.5 * mandelbrot.zoom * cuda.ry) + mandelbrot.moveY;
	newRe = newIm = oldRe = oldIm = 0; //these should start at 0,0

	int i;
	//start the iteration process
	for(i = 0; i < mandelbrot.maxIteration; i++)
	{
		//remember value of previous iterations
		oldRe = newRe;
		oldIm = newIm;
		//the actual iteration, the real and imaginary part are calculated
		newRe = oldRe * oldRe - oldIm * oldIm + pr;
		newIm = 2 * oldRe * oldIm + pi;
		//if the point is outside the circle with radius 2: stop
		if ((newRe * newRe + newIm * newIm) > 8)
			break;
	}
	//use color model conversion to get rainbow palette, make brightness black if maxIterations reached
	//color = HSVtoRGB(ColorHSV(i % 256, 255, 255 * (i < maxIterations)));
	//draw the pixel
	cuda.screen[dim_i] = 0;
	/*
	if (i < mandelbrot.maxIteration)
	{
		int color = 0;
		color = (color << 8) + i & 255;
		color = (color << 8) + 0;
		color = (color << 8) + 0;
		color = (color << 8) + 0xFF;
		cuda.screen[dim_i] = color;
	}
	else if (i % 2 == 0)*/
	if (i < mandelbrot.maxIteration)
		cuda.screen[dim_i] = 0xFFFFFFFF;
}

int
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	static t_mandelbrot	mandelbrot = {{0}, 1, -0.5, 0, 300};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		mandelbrot.moveX -= 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		mandelbrot.moveX += 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		mandelbrot.moveY -= 0.01 / mandelbrot.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		mandelbrot.moveY += 0.01 / mandelbrot.zoom * 10;

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		mandelbrot.zoom += 0.01 * mandelbrot.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		mandelbrot.zoom -= 0.01 * mandelbrot.zoom;

	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		mandelbrot.maxIteration *= 1.1;
		printf("Max iterations = %d\n", mandelbrot.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1 && mandelbrot.maxIteration > 300)
	{
		mandelbrot.maxIteration /= 1.1;
		printf("Max iterations = %d\n", mandelbrot.maxIteration);
	}


	mandelbrot_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, mandelbrot);
	return (0);
}

void
mandelbrot(t_data *data)
{
	do_cuda(data, &mandelbrot_call);
}