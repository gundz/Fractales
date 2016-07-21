#include <header.h>
#include <math.h>
#include <stdlib.h>

void
set_palette(int palette[256])
{
	for (int n = 0; n < 256; n++)
	{
		palette[n] = (int)(n + 512 - 512 * expf(-n / 50.0) / 3.0);
		palette[n] = palette[n] << 24 | palette[n] << 16 | palette[n] << 8 | 255;
	}
	palette[255] = 0;
}

void
burning_ship(t_data *data)
{
	float GraphTop = 1.5f;
	float GraphBottom = -1.5f;
	float GraphLeft = -2.0f;
	float GraphRight = 1.5f;
	int i;
	int max_iteration = 256;

	//ZOOM ZOOM ZINC
	static float zoom = 0.0f;
	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
		zoom += 0.1;
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
		zoom -= 0.1;
	GraphTop -= zoom;
	GraphBottom += zoom;
	GraphLeft += zoom;
	GraphRight -= zoom;

	//MOVE BITCH GET OUT THE WAY
	static float moveX = 0.0f;
	static float moveY = 0.0f;
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_RIGHT) == 1)
		moveX += 0.1;
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_LEFT) == 1)
		moveX -= 0.1;
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_UP) == 1)
		moveY += 0.1;
	if (Esdl_check_input(&data->esdl->en.in, SDL_SCANCODE_DOWN) == 1)
		moveY -= 0.1;
	GraphLeft += moveX;
	GraphRight += moveX;
	GraphTop += moveY;
	GraphBottom += moveY;

	float incrementX = ((GraphRight - GraphLeft) / (SDL_RX - 1));
	float DecrementY = ((GraphTop - GraphBottom) / (SDL_RY - 1));
	float Zx, Zy;
	float CoordReal;
	float CoordImaginary = GraphTop;
	float SquaredX, SquaredY;
	int palette[256];
	set_palette(palette);

	for (int y = 0; y < SDL_RY; y++)
	{
		CoordReal = GraphLeft;
		for (int x = 0; x < SDL_RX; x++)
		{
			i = 0;
			Zx = CoordReal;
			Zy = CoordImaginary;
			SquaredX = Zx * Zx;
			SquaredY = Zy * Zy;
			while ((i < max_iteration) && ((SquaredX + SquaredY) < 4.0))
			{
				Zy = fabs(Zx * Zy);
				Zy = Zy + Zy - CoordImaginary;
				Zx = SquaredX - SquaredY + CoordReal;
				SquaredX = Zx * Zx;
				SquaredY = Zy * Zy;
				i++;
			}
			Esdl_put_pixel(data->surf, x, y, palette[i + 1]);
			CoordReal += incrementX;
		}
		CoordImaginary -= DecrementY;
	}
}