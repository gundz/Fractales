#include <header.h>

void
mandelbrot(t_data *data, SDL_Surface *surf)
{
	//each iteration, it calculates: newz = oldz*oldz + p, where p is the current pixel, and oldz stars at the origin
	double pr, pi;           //real and imaginary part of the pixel p
	double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old z
	double zoom = 1, moveX = -0.5, moveY = 0; //you can change these to zoom and change position
	int maxIterations = 300;//after how much iterations the function should stop

	for (int y = 0; y < SDL_RY; y++)
	{
		for (int x = 0; x < SDL_RX; x++)
		{
			//calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
			pr = 1.5 * (x - SDL_RX / 2) / (0.5 * zoom * SDL_RX) + moveX;
			pi = (y - SDL_RY / 2) / (0.5 * zoom * SDL_RY) + moveY;
			newRe = newIm = oldRe = oldIm = 0; //these should start at 0,0
			//"i" will represent the number of iterations
			int i;
			//start the iteration process
			for(i = 0; i < maxIterations; i++)
			{
				//remember value of previous iteration
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

			if (i < maxIterations)
				Esdl_put_pixel(surf, x, y, 0xFFFFFFFF);
		}
	}
}