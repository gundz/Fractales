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
	return (get_num_groups(0) * get_local_size((0)));
}

int
getGridHeight(void)
{
	return (get_num_groups(0) * get_local_size((0)));
}

int
getIndex(void)
{
	return (getIndexY() * getGridWidth() + getIndexX());
}

__kernel
void
mandelbrot(__global int *output)
{
	int					col = getIndexX();
	int					row = getIndexY();
	int					width = getGridWidth();
	int					height = getGridHeight();

	double				c_re = (col - width / 2.0) * 4.0 / width;
	double				c_im = (row - height / 2.0) * 4.0 / width;
	double				x = 0;
	double				y = 0;
	int					it = 0;

	int					max_it = 100;

	while (x * x + y * y <= 4 && it < max_it)
	{
		double x_new = x * x - y * y + c_re;
		y = 2 * x * y + c_im;
		x = x_new;
		it++;
		if (it < max_it / 2)
		{
			output[getIndex()] = (0x000000FF << 16) + (it % 255);
			output[getIndex()] = (0xFF0000FF << 8) + (it % 255);
		}
		else if (it < max_it)
		{
			output[getIndex()] = (0x000000FF << 8) + (it % 255);
			output[getIndex()] = (0x000000FF << 8) + (it % 255);
		}
		else
			output[getIndex()] = 0xFFFFFFFF;
	}
}

