#include <header.h>
#include <math.h>
#include <stdlib.h>

void
burning_ship(t_data *data)
{
	float GraphTop = 1.5f;
	float GraphBottom = -1.5f;
	float GraphLeft = -2.0f;
	float GraphRight = 1.5f;
	int i;
	int max_iteration = 256;

	float incrementX = ((GraphRight - GraphLeft) / (SDL_RX - 1));
	float DecrementY = ((GraphTop - GraphBottom) / (SDL_RY - 1));
	float Zx, Zy;
	float CoordReal;
	float CoordImaginary = GraphTop;
	float SquaredX, SquaredY;
	int palette[256];

	for (int n = 0; n < 256; n++)
	{
		palette[n] = (int)(n + 512 - 512 * expf(-n / 50.0) / 3.0);
		palette[n] = palette[n] << 24 | palette[n] << 16 | palette[n] << 8 | 255;
	}
	palette[255] = 0;

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
			do
			{
				Zy = fabs(Zx * Zy);
				Zy = Zy + Zy - CoordImaginary;
				Zx = SquaredX - SquaredY + CoordReal;
				SquaredX = Zx * Zx;
				SquaredY = Zy * Zy;
				i++;
			} while ((i < max_iteration) && ((SquaredX + SquaredY) < 4.0));
			i--;

			Esdl_put_pixel(data->surf, x, y, palette[i]);
			CoordReal += incrementX;
		}
		CoordImaginary -= DecrementY;
	}
}