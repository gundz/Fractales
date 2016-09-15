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

	double pr, pi;
	double newRe, newIm;
	double oldRe, oldIm;
	int i;

	// pr = (x * (cuda.rx / cuda.ry) - cuda.rx * 0.5) * burning.zoom + burning.moveX;
	// pi = (y - cuda.ry * 0.5) * burning.zoom + burning.moveY;
	pr = (x - cuda.rx / 2) / (0.5 * burning.zoom * cuda.rx) + burning.moveX;
	pi = (y - cuda.ry / 2) / (0.5 * burning.zoom * cuda.ry) + burning.moveY;
	oldRe = pr;
	oldIm = pi;
	i = 0;
	while (i < burning.maxIteration)
	{
		newRe = (oldRe * oldRe - oldIm * oldIm) + pr;
		newIm = (fabs(oldRe * oldIm) * 2) + pi;
		if ((newRe * newRe + newIm * newIm) > 4)
			break ;
		oldRe = newRe;
		oldIm = newIm;
		i++;
	}

    if(i == burning.maxIteration)
        cuda.screen[dim_i] = 0x00000000;
    else
    {
        double z = sqrt(newRe * newRe + newIm * newIm);
        int brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2(double(burning.maxIteration));
        cuda.screen[dim_i] = brightness << 24 | (i % 255) << 16 | 255 << 8 | 255;
    }
}

int
burning_ship_call(t_data *data, t_cuda *cuda)
{
	static t_burning_ship burning = {1, 0, 0, 200};

	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		burning.moveX -= 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		burning.moveX += 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		burning.moveY -= 0.01 / burning.zoom * 10;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		burning.moveY += 0.01 / burning.zoom * 10;

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		burning.zoom += 0.01 * burning.zoom;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		burning.zoom -= 0.01 * burning.zoom;

	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
	{
		burning.maxIteration *= 1.1;
		printf("Max iterations = %d\n", burning.maxIteration);
	}
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
	{
		burning.maxIteration *= 0.9;
		printf("Max iterations = %d\n", burning.maxIteration);
	}

	burning_ship_kernel<<<cuda->gridSize, cuda->blockSize>>>(*cuda, burning);
	return (0);
}

void
burning_ship(t_data *data)
{
	do_cuda(data, &burning_ship_call);
}