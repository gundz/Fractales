#ifndef MANDELBROT_H
# define MANDELBROT_H

typedef struct	s_mandelbrot
{
	double		zoom;
	double		moveX;
	double		moveY;
	int			maxIteration;
	int			palette[256];
}				t_mandelbrot;

#endif