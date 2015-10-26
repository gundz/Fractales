/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   libft.h                                            :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <fgundlac@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2013/11/19 10:51:44 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/01 23:40:20 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LIBFT_H
# define LIBFT_H

# include <get_next_line.h>
# include <lst.h>
# include <stddef.h>

void			ft_putchar(char c);
void			ft_putstr(const char *const str);
void			ft_putendl(const char *const str);
void			ft_putnbr(int nbr);
void			ft_putchar_fd(char c, int fd);
void			ft_putstr_fd(const char *const str, int fd);
void			ft_putendl_fd(const char *const str, int fd);
void			ft_putnbr_fd(int nbr, const int fd);

char			*ft_strcpy(char *dest, const char *src);
char			*ft_strncpy(char *dest, char *src, size_t n);

size_t			ft_strlen(const char *str);
int				ft_strcmp(const char *s1, const char *s2);

int				ft_strncmp(const char *s1, const char *s2, size_t n);
char			*ft_strstr(const char *s1, const char *s2);
char			*ft_strnstr(const char *s1, const char *s2, size_t n);

char			*ft_strdup(const char *const str);
char			*ft_strcat(char *dest, const char *const src);
char			*ft_strjoin(char *str, char *strc);
char			*ft_strijoin(unsigned int n, ...);
char			*ft_strsub(char const *s, unsigned int start, size_t len);
char			**ft_strsplit(char *str, char c);
char			*ft_itoa(int n);
char			*ft_strrev(char *str);
char			*ft_strchr(const char *src, int c);

int				ft_atoi(const char *str);

unsigned int	ft_ctablen(char **const tab);
void			ft_freectab(char **tab);

char			*ft_basename(char *path);

#endif
