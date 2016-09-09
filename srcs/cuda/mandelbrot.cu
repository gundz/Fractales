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
		if ((newRe * newRe + newIm * newIm) > 4)
			break;
	}
	cuda.screen[dim_i] = mandelbrot.palette[(i + 1) % 255];
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
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	static t_mandelbrot	mandelbrot = {1, -0.5, 0, 300, {0}};

	set_palette(mandelbrot.palette);

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