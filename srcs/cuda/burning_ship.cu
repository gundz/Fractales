extern "C"
{
#include <header.h>
}
#include <cuda.h>
#include <burning_ship.h>

__global__ void
burning_ship_kernel(t_cuda cuda, t_burning_ship burning)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int dim_i = y * cuda.rx + x;
	if ((x >= cuda.rx) || (y >= cuda.ry))
		return ;

	float GraphTop = 1.5f;
	float GraphBottom = -1.5f;
	float GraphLeft = -2.0f;
	float GraphRight = 1.5f;
	int i;
	int max_iteration = 256;

	float incrementX = ((GraphRight - GraphLeft) / (cuda.rx - 1));
	float DecrementY = ((GraphTop - GraphBottom) / (cuda.ry - 1));
	float Zx, Zy;
	float CoordReal;
	float CoordImaginary = GraphTop;
	float SquaredX, SquaredY;

	CoordReal = GraphLeft + (incrementX * x);
	CoordImaginary = GraphTop - (DecrementY * y);
	i = 0;
	Zx = CoordReal;
	Zy = CoordImaginary;
	SquaredX = Zx * Zx;
	SquaredY = Zy * Zy;

	cuda.screen[dim_i] = 0;
	while ((i < max_iteration) && ((SquaredX + SquaredY) < 4.0))
	{
		Zy = fabs(Zx * Zy);
		Zy = Zy + Zy - CoordImaginary;
		Zx = SquaredX - SquaredY + CoordReal;
		SquaredX = Zx * Zx;
		SquaredY = Zy * Zy;
		i++;
	}
	cuda.screen[dim_i] = burning.palette[(i + 1) % 255];
}

static void
set_palette(int palette[256])
{
	for (int n = 0; n < 256; n++)
	{
		palette[n] = (int)(n + 512 - 512 * expf(-n / 50.0) / 3.0);
		palette[n] = palette[n] << 24 | palette[n] << 16 | palette[n] << 8 | 255;
	}
	palette[255] = 0;
}

int
burning_ship_call(t_data *data, t_cuda *cuda)
{
	t_burning_ship burning;
	set_palette(burning.palette);

	burning_ship_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, burning);
	return (0);
}

void
burning_ship(t_data *data)
{
	do_cuda(data, &burning_ship_call);
}