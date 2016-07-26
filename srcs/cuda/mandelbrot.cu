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
	double zoom = 1, moveX = -0.5, moveY = 0; //you can change these to zoom and change position
	int maxIterations = 300;//after how much iterations the function should stop

	//calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
	pr = 1.5 * (x - cuda.rx / 2) / (0.5 * zoom * cuda.rx) + moveX;
	pi = (y - cuda.ry / 2) / (0.5 * zoom * cuda.ry) + moveY;
	newRe = newIm = oldRe = oldIm = 0; //these should start at 0,0

	int i;
	//start the iteration process
	for(i = 0; i < maxIterations; i++)
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
	//use color model conversion to get rainbow palette, make brightness black if maxIterations reached
	//color = HSVtoRGB(ColorHSV(i % 256, 255, 255 * (i < maxIterations)));
	//draw the pixel
	cuda.screen[dim_i] = 0;
	if (i < maxIterations)
	{
		int color = 0;
		color = (color << 8) + i & 255;
		color = (color << 8) + 0;
		color = (color << 8) + 0;
		color = (color << 8) + 0xFF;
		cuda.screen[dim_i] = color;
	}
	else if (i % 2 == 0)
		cuda.screen[dim_i] = 0xFFFFFFFF;
}

int
mandelbrot_call(t_data *data, t_cuda *cuda)
{
	t_mandelbrot mandelbrot = {};
	mandelbrot_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, mandelbrot);
	return (0);
}

void
mandelbrot(t_data *data)
{
	do_cuda(data, &mandelbrot_call);
}