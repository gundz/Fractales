#ifndef JULIA_H
# define JULIA_H

typedef struct	s_julia
{
	int			mx;
	int			my;
	double		zoom;
	double		moveX;
	double		moveY;
	int			maxIteration;
	int			palette[256];
}				t_julia;

#endif