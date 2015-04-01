/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   mlx_lib.h                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/01 01:08:47 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 01:09:04 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef MLX_LIB_H
# define MLX_LIB_H

# include <mlx_const.h>

# define RX		640
# define RY		480

typedef struct	s_mlx_surf
{
	char		*pixels;
	void		*img;
	int			x;
	int			y;
	int			bpp;
	int			pitch;
	int			endian;
}				t_mlx_surf;

typedef struct	s_mlx
{
	t_mlx_surf	*surf;
	void		*mlx;
	void		*win;
	int			m_x;
	int			m_y;
}				t_mlx;

typedef struct	s_data
{
	t_mlx		mlx;
}				t_data;

unsigned int	mlx_l_init(t_mlx *mlx, char *title, const int x, const int y);
int				mlx_quit(unsigned int key, t_mlx *mlx);

t_mlx_surf		*mlx_create_rgb_surface(void *const mlx,
		const int x, const int y, const int color);
void			mlx_free_surface(void *mlx, t_mlx_surf *surf);
void			mlx_show_surf(t_mlx *mlx, t_mlx_surf *surf);

void			put_pixel(t_mlx_surf *surf, const int x, const int y,
		const int color);

int				mlx_k_event(unsigned int key, t_data *param);
int				mlx_m_event(unsigned int button, int x, int y, t_data *param);

#endif
