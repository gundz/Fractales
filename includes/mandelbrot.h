#ifndef MANDELBROT_H
# define MANDELBROT_H

typedef struct	s_mandelbrot
{
	int			palette[256];
	double		zoom;
	double		moveX;
	double		moveY;
	int			maxIteration;
}				t_mandelbrot;

#endif