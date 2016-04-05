int
getIndexX(void)
{
	return (get_global_id(0));
}

int
getIndexY(void)
{
	return (get_global_id(1));
}

int
getGridWidth(void)
{
	return (get_global_size(0));
}

int
getGridHeight(void)
{
	return (get_global_size(1));
}

int
getIndex(void)
{
	return (getIndexY() * getGridWidth() + getIndexX());
}

typedef struct			s_fdata
{
	float				x;
	float				y;
	double				scale;
	int					max_it;
}						t_fdata;

__kernel void
mandelbrot(__global unsigned int *output, __global t_fdata *input)
{
	int					col = getIndexX();
	int					row = getIndexY();
	int					width = getGridWidth();
	int					height = getGridHeight();

	t_fdata fdata = *input;
	double scale = fdata.scale;
	double ax = fdata.x;
	double ay = fdata.y;
	double max_it = fdata.max_it;
	int iter = 0;

	double				cx = (col - width / 2.0) * scale + ax;
	double				cy = (row - height / 2.0) * scale * + ay;

	double				x, y;
	double				zx, zy;
	double				zx2, zy2;
	double				x_new;

	zx2 = 0;
	zy2 = 0;
	x = 0;
	y = 0;


	while (iter < max_it && (zx2 + zy2) < 4)
	{
		x_new = (zx2 - zy2) + cx;
		y = 2 * x * y + cy;
		x = x_new;

		zx2 = x * x;
		zy2 = y * y;

		iter++;		
		if (iter < max_it / 2)
		{
			output[getIndex()] = (0x000000FF << 16) + (iter % 255);
			output[getIndex()] = (0xFF0000FF << 8) + (iter % 255);
		}
		else if (iter < max_it)
		{
			output[getIndex()] = (0x000000FF << 8) + (iter % 255);
			output[getIndex()] = (0x000000FF << 8) + (iter % 255);
		}
		else
			output[getIndex()] = 0xFFFFFFFF;
	}
}
