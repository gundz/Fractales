#TYPE: LIB or PROGRAM
TYPE =			LIB

NAME =			libft.a

SRCS =			\
				ft_strlen.c \
				ft_strcmp.c \
				ft_strncmp.c \
				ft_strstr.c \
				ft_strnstr.c \
				\
				ft_putchar.c \
				ft_putstr.c \
				ft_putendl.c \
				ft_putnbr.c \
				ft_putchar_fd.c \
				ft_putstr_fd.c \
				ft_putendl_fd.c \
				ft_putnbr_fd.c \
				\
				ft_strcpy.c \
				ft_strncpy.c \
				ft_strdup.c \
				ft_strcat.c \
				ft_strjoin.c \
				ft_strijoin.c \
				ft_strsub.c \
				ft_strsplit.c \
				ft_itoa.c \
				ft_strrev.c \
				ft_strchr.c \
				\
				ft_atoi.c \
				\
				ft_ctablen.c \
				ft_freectab.c \

SRCS +=			\
				lst_create_elem.c \
				lst_push_back.c \
				lst_csize.c \
				lst_count.c \
				lst_b_sort.c \
				lst_free.c \
				lst_to_char.c \

SRCS +=			\
				thread.c \
				t_tab_thread.c \

SRCS +=			\
				get_next_line.c \

SRCS +=			\
				ft_basename.c \

CC =			gcc

CFLAGS =		-Wall -Werror -Wextra -g

#LIB_NAMES =		-lfoo -lbar
#LIB_PATH =		./libfoo/ ./libbar/

#LIB_SUPP =		-lm

SRC_PATH = 		./srcs/
INC_PATH = 		./includes/
OBJ_PATH =		./obj/

OBJ_NAME = $(SRCS:.c=.o)
SRC = $(addprefix $(SRC_PATH), $(SRCS))
OBJ = $(addprefix $(OBJ_PATH), $(OBJ_NAME))
LIB = $(addprefix -I, $(addsuffix includes, $(LIB_PATH))) $(addprefix -L, $(LIB_PATH))
INC = $(addprefix -I, $(INC_PATH))
LDFLAGS = $(LIB) $(LIB_NAMES)
EMPTY =

all: libs name $(OBJ) done $(NAME)

name :
	@ echo "\033[4;7mCompiling Objects:\033[0m [$(NAME)]"

done :
	@ echo "\n\033[4m\033[95md\033[93mo\033[32mn\033[96me\033[0m \033[91m!\033[0m\n"

$(NAME): $(OBJ)
	@ echo "\033[4;7mCompiling binary:\033[0m [$(NAME)]"
ifeq ($(TYPE), LIB)
	@ ar -rc $(NAME) $(OBJ)
	@ ranlib $(NAME)
else
	@ $(CC) $(OBJ) $(LDFLAGS) $(LIB_SUPP) -o $(NAME)
endif
	@ echo "\033[4m\033[95md\033[93mo\033[32mn\033[96me\033[0m \033[91m!\033[0m\n"

$(OBJ_PATH)%.o: $(SRC_PATH)%.c
	@ printf "\033[37;7m+\033[0m"
	@ mkdir -p $(OBJ_PATH) 2> /dev/null
	@ $(CC) $(CFLAGS) $(LIB) $(INC) -c $< -o $@

ifeq ($(LIB_PATH), $(EMPTY))
else
libs:
	@ $(foreach lib, $(LIB_PATH), make -sC $(lib);)
endif

clean:
	@ rm -rf $(OBJ_PATH)
	@ echo "\033[4;7mCleaning:\033[0m [$(NAME)]\n\033[4m\033[95md\033[93mo\033[32mn\033[96me\033[0m \033[91m!\033[0m\n"
ifeq ($(LIB_PATH), $(EMPTY))
else
	@ $(foreach lib, $(LIB_PATH), make clean -sC $(lib);)
endif

fclean: clean
	@ rm -f $(NAME)
	@ echo "\033[4;7mFcleaning:\033[0m [$(NAME)]\033[0m\n\033[4m\033[95md\033[93mo\033[32mn\033[96me\033[0m \033[91m!\033[0m\n"
ifeq ($(LIB_PATH), $(EMPTY))
else
	@ $(foreach lib, $(LIB_PATH), make fclean -sC $(lib);)
endif

re: fclean all

.PHONY: name clean fclean libs
