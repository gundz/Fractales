#include <math.h>

int				color_it(double re, double im, int i, int max_iteration)
{
	double		z;
	int			brightness;

	z = sqrt(re * re + im * im);
	brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
	return (brightness);
}