static __device__ int
hsv_to_rgb(float h, float s, float v)
{
    float r, g, b; //this function works with floats between 0 and 1
    h = h / 256.0;
    s = s / 256.0;
    v = v / 256.0;
     //If saturation is 0, the color is a shade of gray
    if(s == 0)
        r = g = b = v;
        //If saturation > 0, more complex calculations are needed
    else
    {
        float f, p, q, t;
        int i;
        h *= 6; //to bring hue to a number between 0 and 6, better for the calculations
        i = (int)floor(h);  //e.g. 2.7 becomes 2 and 3.01 becomes 3 or 4.9999 becomes 4
        f = h - i;  //the fractional part of h
        p = v * (1 - s);
        q = v * (1 - (s * f));
        t = v * (1 - (s * (1 - f)));
        switch(i)
        {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5: r = v; g = p; b = q; break;
        }
    }
    return (int)(r * 255.0) << 24 | (int)(g * 255.0) << 16 | (int)(b * 255.0) << 8 | 255;
}

static __device__ int
cuda_color_it(double new_re, double new_im, int i, int max_iteration)
{
    double      z;
    int         brightness;

    z = sqrt(new_re * new_re + new_im * new_im);
    brightness = 256. * log2(1.75 + i - log2(log2(z))) / log2((double)(max_iteration));
    return brightness;
}