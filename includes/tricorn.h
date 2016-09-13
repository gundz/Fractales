#ifndef TRICORN_H
# define TRICORN_H

typedef struct	s_tricorn
{
	double		zoom;
	double		moveX;
	double		moveY;
	int			maxIteration;
	int			palette[256];
}				t_tricorn;

#endif