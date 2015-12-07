#include <easy_sdl.h>
#include <OCL.h>

#define RX 1920
#define RY 1080

typedef struct			s_data
{
	t_esdl				*esdl;
}						t_data;

void
compute(SDL_Surface *surf, t_cl_data *cl_data)
{
	#define				NUM_ELEMENTS_X RX
	#define				NUM_ELEMENTS_Y RY
	#define				NUM_ELEMENTS (NUM_ELEMENTS_X * NUM_ELEMENTS_Y)

	int					tab[NUM_ELEMENTS];
	//cl_mem				input;
	cl_mem				output;

	size_t				i, j;
	// for (i = 0; i < NUM_ELEMENTS; i++)
	// {
	// 	tab[i] = 0;
	// }

	//input = clCreateBuffer(cl_data.context, CL_MEM_READ_ONLY, sizeof(*tab) * NUM_ELEMENTS, NULL, NULL);
	output = clCreateBuffer(cl_data->context, CL_MEM_WRITE_ONLY, sizeof(*tab) * NUM_ELEMENTS, NULL, NULL);

	//clEnqueueWriteBuffer(cl_data.command_queue, input, CL_TRUE, 0, sizeof(*tab) * NUM_ELEMENTS, tab, 0, NULL, NULL);

	//clSetKernelArg(cl_data.kernel, 0, sizeof(cl_mem), &input);
	clSetKernelArg(cl_data->kernel, 0, sizeof(cl_mem), &output);

	size_t				global_item_size[2] = {NUM_ELEMENTS_X, NUM_ELEMENTS_Y};
    // size_t				local_item_size[2] = {4, 4};

	clEnqueueNDRangeKernel(cl_data->command_queue, cl_data->kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
	clFinish(cl_data->command_queue);

	clEnqueueReadBuffer(cl_data->command_queue, output, CL_TRUE, 0, sizeof(*tab) * NUM_ELEMENTS, tab, 0, NULL, NULL);

	// for (i = 0; i < NUM_ELEMENTS_Y; i++)
	// {
	// 	printf("%zu :\n", i);
	// 	for (j = 0; j < NUM_ELEMENTS_X; j++)
	// 		printf("%f | ", tab[i * NUM_ELEMENTS_X + j]);
	// 	printf("\n");
	// }

	for (i = 0; i < NUM_ELEMENTS_Y; i++)
	{
		for (j = 0; j < NUM_ELEMENTS_X; j++)
			Esdl_put_pixel(surf, j, i, tab[i * NUM_ELEMENTS_X + j]);
	}

	//clReleaseMemObject(input);
	clReleaseMemObject(output);
}

void					test(t_data *data, t_cl_data *cl_data)
{
	SDL_Surface			*surf;
	SDL_Texture			*tex;

	surf = Esdl_create_surface(RX, RY);

	compute(surf, cl_data);

	tex = SDL_CreateTextureFromSurface(data->esdl->en.ren, surf);
	SDL_FreeSurface(surf);

	SDL_RenderClear(data->esdl->en.ren);
	SDL_RenderCopy(data->esdl->en.ren, tex, NULL, NULL);
	SDL_DestroyTexture(tex);
	SDL_RenderPresent(data->esdl->en.ren);
}

void
init(t_cl_data *cl_data)
{
	if (OCLInit(cl_data) == CL_SUCCESS)
	{
		if (OCLBuildPRogram(cl_data, "srcs/test.cl") == CL_SUCCESS)
		{
			if (OCLCreateKernel(cl_data, "mandelbrot") == CL_SUCCESS)
			{
			}
		}
	}
}

int					main(int argc, char **argv)
{
	t_data			data;
	t_esdl			esdl;
	t_cl_data		cl_data;

	data.esdl = &esdl;

	if (Esdl_init(&esdl, RX, RY, 60, "Engine") == -1)
		return (-1);
	init(&cl_data);
	while (esdl.run)
	{
		Esdl_update_events(&esdl.en.in, &esdl.run);

		test(&data, &cl_data);

		Esdl_fps_limit(&esdl);
		Esdl_fps_counter(&esdl);
	}


	clReleaseKernel(cl_data.kernel);
	clReleaseProgram(cl_data.program);
	clReleaseCommandQueue(cl_data.command_queue);
	clReleaseContext(cl_data.context);	

	Esdl_quit(&esdl);
	(void)argc;
	(void)argv;
	return (0);
}