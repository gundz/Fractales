#include <easy_sdl.h>
#include <OCL.h>

#include <sys/time.h>

#define RX 640
#define RY 480
#define SIZEBITCH RX * RY * 4

typedef struct			s_data
{
	t_esdl				*esdl;
}						t_data;

typedef struct			s_fdata
{
	float				x;
	float				y;
	double				scale;
	int					max_it;
}						t_fdata;

void
compute(t_data *data, SDL_Surface *surf, t_cl_data *cl_data)
{
	static t_fdata fdata = {-.6, 0, 1./256, 128};

	if (data->esdl->en.in.button[SDL_BUTTON_LEFT] == 1)
	{
		fdata.max_it *= 1.05;

		fdata.x += (data->esdl->en.in.m_x - RX / 2) * fdata.scale;
		fdata.y += (data->esdl->en.in.m_y - RY / 2) * fdata.scale;

		if (fdata.scale > fabs(fdata.x) * 1e-16 && fdata.scale > fabs(fdata.y) * 1e-16)
			fdata.scale /= 2;

		printf("max_it = %d\n", fdata.max_it);
		data->esdl->en.in.button[SDL_BUTTON_LEFT] = 0;
	}
	if (data->esdl->en.in.button[SDL_BUTTON_RIGHT] == 1)
	{
		fdata.x += (data->esdl->en.in.m_x - RX / 2) * fdata.scale;
		fdata.y += (data->esdl->en.in.m_y - RY / 2) * fdata.scale;

		fdata.scale *= 2;
		data->esdl->en.in.button[SDL_BUTTON_RIGHT] = 0;
	}
	

	clSetKernelArg(cl_data->kernel, 0, sizeof(cl_mem), &(cl_data->output));
	clSetKernelArg(cl_data->kernel, 1, sizeof(cl_mem), &(cl_data->input));

	size_t					global_item_size[2] = {surf->w, surf->h};

    clEnqueueWriteBuffer(cl_data->command_queue, cl_data->input, CL_TRUE, 0, sizeof(t_fdata), &fdata, 0, NULL, NULL);

	clEnqueueNDRangeKernel(cl_data->command_queue, cl_data->kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);

	clFinish(cl_data->command_queue);

	clEnqueueReadBuffer(cl_data->command_queue, cl_data->output, CL_TRUE, 0, SIZEBITCH, surf->pixels, 0, NULL, NULL);
}

void					test(t_data *data, t_cl_data *cl_data)
{
	SDL_Surface			*surf;
	SDL_Texture			*tex;

	surf = Esdl_create_surface(RX, RY);

	compute(data, surf, cl_data);

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


	cl_data->output = clCreateBuffer(cl_data->context, CL_MEM_READ_WRITE, SIZEBITCH, NULL, NULL);

	cl_data->input = clCreateBuffer(cl_data->context, CL_MEM_READ_WRITE, sizeof(t_fdata), NULL, NULL);
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

	clReleaseMemObject(cl_data.output);
	clReleaseKernel(cl_data.kernel);
	clReleaseProgram(cl_data.program);
	clReleaseCommandQueue(cl_data.command_queue);
	clReleaseContext(cl_data.context);	

	Esdl_quit(&esdl);
	(void)argc;
	(void)argv;
	return (0);
}
