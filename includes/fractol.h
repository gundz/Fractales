/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   fractol.h                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 04:58:18 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 04:58:18 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef FRACTOL_H
# define FRACTOL_H

# define NB_THREAD 6
# define ZOOM 1.1
# define RXZ RX / ZOOM
# define RYZ RY / ZOOM
# define NB_FRACT 3

# define NM_COLOR 4  * 256

# include <thread.h>
# include <mlx_lib.h>

typedef struct		s_v2d
{
	long double		x;
	long double		y;
}					t_v2d;

typedef struct		s_fract
{
	t_mlx_surf		*surf;
	int				i;
	int				max_it;
	long double		zoom;
	t_v2d			zoomp;
	t_v2d			coor;
	t_v2d			pos;
	double			pos_inc;
	int				c_map[NM_COLOR];
}					t_fract;

typedef struct		s_fracts
{
	t_fract			data;
	void			(*init)(t_fract *data);
	void			*(*fract)(void *arg, const int x, const int);
	int				(*input)();
	int				thread_status;
}					t_fracts;

typedef struct		s_data
{
	t_fracts		*fract;
	t_fracts		fracts[NB_FRACT];
	int				fract_i;
	t_mlx			mlx;
	t_thread		*thread;
	t_tab			*tab;
}					t_data;

int					main_mlx(t_data *data);

void				*mandelbrot(void *arg, const int x, const int y);
void				init_mandelbrot(t_fract *data);
int					mandelbrot_input(unsigned int button, unsigned int key,
	t_data *data);

void				*julia(void *arg, const int x, const int y);
void				init_julia(t_fract *data);
int					julia_input(unsigned int button, unsigned int key,
	t_data *data);

void				*sierpinski(void *arg, const int x, const int y);
void				init_sierpinski(t_fract *data);
int					sierpinski_input(unsigned int button, unsigned int key,
	t_data *data);

t_v2d				set_v2d(long double x, long double y, t_v2d *v);
t_v2i				set_v2i(int x, int y, t_v2i *v);

int					change_fract(t_data *data);

#endif
