#ifndef CUDA_H
# define CUDA_H

typedef struct	s_cuda
{
	Uint32		*screen; //screen pointer
	int			rx; //size x of screen
	int			ry; //size y of screen
	dim3		blockSize;
	dim3		gridSize; //CUDA gridSize
	int			bx;
	int			by;
}				t_cuda;

void			do_cuda(t_data *data, int (*f)(t_data *, t_cuda *));

#endif