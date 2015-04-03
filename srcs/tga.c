/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   tga.c                                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/04/02 08:50:50 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/03 13:04:33 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <mlx_lib.h>
#include <unistd.h>
#include <fcntl.h>
#include <math.h>

static void		ft_putchar_fd(const char c, const int fd)
{
	write(fd, &c, 1);
}

static int		save_file(const char *const name)
{
	int			fd;

	if ((fd = open(name,
		O_WRONLY | O_CREAT, S_IRUSR | S_IRGRP | S_IROTH)) == -1)
		return (-1);
	return (fd);
}

static void		write_tga_header(const int fd, const int w, const int h)
{
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(2, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd(0, fd);
	ft_putchar_fd((w & 0x00ff), fd);
	ft_putchar_fd((w & 0xff00) / 256, fd);
	ft_putchar_fd((h & 0x00ff), fd);
	ft_putchar_fd((h & 0xff00) / 256, fd);
	ft_putchar_fd(24, fd);
	ft_putchar_fd(0x00, fd);
}

static void		write_data(const int fd, unsigned int *img,
	const unsigned int w, const unsigned int h)
{
	unsigned int	i;
	float			big;
	float			small;
	float			ramp;

	big = 0;
	i = 0;
	while (i < w)
		big = MAX(big, img[i++]);
	small = big;
	i = 0;
	while (i < h)
		small = MIN(small, img[i++]);
	i = 0;
	while (i < w * h)
	{
		ramp = 2 * (img[i] - small) / (big - small);
		if (ramp > 1)
			ramp = 1;
		ramp = pow(ramp, 0.5);
		ft_putchar_fd((int)(ramp * 255), fd);
		ft_putchar_fd((int)(ramp * 255), fd);
		ft_putchar_fd((int)(ramp * 255), fd);
		i++;
	}
}

void			write_tga(char *name, unsigned int *img,
	const int w, const int h)
{
	int			fd;

	if ((fd = save_file(name)) == -1)
		return ;
	write_tga_header(fd, w, h);
	write_data(fd, img, w, h);
	close(fd);
}
