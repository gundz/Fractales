#include <header.h>
#include <math.h>

int						color_it(double re, double im, int i, int max_iteration)
{
	double				z;
	int					brightness;

	z = sqrt(re * re + im * im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness);
}

void					fractal_zoom(t_data *data, t_fractal *fractal)
{
	fractal->oldcx = fractal->cx;
	fractal->oldcy = fractal->cy;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		fractal->zoom = fractal->zoom / 1.05;
		fractal->cx = (fractal->oldcx) + (fractal->mx * 0.05) * fractal->zoom;
		fractal->cy = (fractal->oldcy) + (fractal->my * 0.05) * fractal->zoom;
		fractal->maxiteration *= 1.0025;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		fractal->zoom = fractal->zoom * 1.05;
		fractal->cx = (fractal->oldcx) + (fractal->mx * 0.05) * fractal->zoom;
		fractal->cy = (fractal->oldcy) + (fractal->my * 0.05) * fractal->zoom;
		fractal->maxiteration *= 0.9975;
	}
}

void					fractal_input(t_data *data, t_fractal *fractal)
{
	fractal_zoom(data, fractal);
	if (data->esdl->en.in.key[SDL_SCANCODE_LEFT] == 1)
		fractal->movex -= 10 * fractal->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_RIGHT] == 1)
		fractal->movex += 10 *fractal->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_UP] == 1)
		fractal->movey -= 10 *fractal->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_DOWN] == 1)
		fractal->movey += 10 *fractal->zoom;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_PLUS] == 1)
		fractal->maxiteration *= 1.1;
	if (data->esdl->en.in.key[SDL_SCANCODE_KP_MINUS] == 1)
		fractal->maxiteration *= 0.9;
}