# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2015/04/01 01:27:56 by fgundlac          #+#    #+#              #
#    Updated: 2015/04/03 19:15:30 by fgundlac         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

NAME = project

CC = ~/.brew/bin/gcc-4.9
#CC = gcc
#CFLAGS = -Wall -Wextra -Werror -g
CFLAGS = -Wall -Wextra -Werror -Ofast
CFLAGS += -lmlx -framework OpenGL -framework AppKit

PATH_INC = includes/
PATH_OBJ = obj
PATH_SRC = srcs

SRC =	main.c \
		mlx_init.c \
		mlx_surface.c \
		mlx_draw.c \
		mlx_k_event.c \
		mlx_m_event.c \
		mandelbrot.c \
		julia.c \
		julia_input.c \
		mandelbrot_input.c \
		tga_tools.c \
		tga.c \
		tools.c \

LIBPATH = libft/
LIBINC = $(LIBPATH)includes
LIBLIB = $(LIBPATH)libft.a

OBJ = $(patsubst %.c, $(PATH_OBJ)/%.o, $(SRC))

all: lib $(NAME)

$(NAME): namemes $(OBJ)
	@ $(CC) $(OBJ) $(CFLAGS) -I $(PATH_INC) -I $(LIBINC) $(LIBLIB) -o $(NAME)
	@ echo " \033[4m\033[95md\033[93mo\033[32mn\033[96me \033[91m!\033[0m"

$(PATH_OBJ)/%.o: $(addprefix $(PATH_SRC)/, %.c)
	@ echo -n .
	@ mkdir -p $(PATH_OBJ)
	@ $(CC) -c $^ -I $(PATH_INC) $(CFLAGS) -I $(LIBINC) -o $@

lib:
	@ make -C $(LIBPATH)

clean:
	@ make clean -C $(LIBPATH)
	@ rm -rf $(PATH_OBJ)
	@ echo "Cleaning $(NAME) \
		\033[4m\033[95md\033[93mo\033[32mn\033[96me \033[91m!\033[0m"

fclean: clean
	@ make fclean -C $(LIBPATH)
	@ rm -rf $(NAME)
	@ echo "Fcleaning $(NAME) \
		\033[4m\033[95md\033[93mo\033[32mn\033[96me \033[91m!\033[0m"

namemes :
	@ echo -n Compiling $(NAME)

re: fclean all

.PHONY: clean fclean re
