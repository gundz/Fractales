/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   cuda.h                                             :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2016/09/17 15:59:32 by fgundlac          #+#    #+#             */
/*   Updated: 2016/09/17 15:59:33 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef CUDA_H
# define CUDA_H

typedef struct	s_cuda
{
	Uint32		*screen;
	int			rx;
	int			ry;
	dim3		blocksize;
	dim3		gridsize;
	int			bx;
	int			by;
}				t_cuda;

void			do_cuda(t_data *data, int (*f)(t_data *, t_cuda *));

#endif
