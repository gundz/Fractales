#ifndef BURNING_SHIP_H
# define BURNING_SHIP_H

typedef struct	s_burning_ship
{
	double		zoom;
	double		moveX;
	double		moveY;
	int			maxIteration;
	int			palette[256];
}				t_burning_ship;

#endif