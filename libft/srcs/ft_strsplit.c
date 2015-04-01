/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_strsplit.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2014/11/11 18:28:26 by fgundlac          #+#    #+#             */
/*   Updated: 2014/11/11 20:04:45 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#include <libft.h>
#include <stdlib.h>

static int				count_words(char *str, char c)
{
	int					nb_words;
	int					i;

	nb_words = 0;
	i = 0;
	while (str[i] != '\0')
	{
		while (str[i] == c)
			i++;
		if (str[i] == '\0')
			return (nb_words);
		nb_words++;
		while (str[i] != c && str[i] != '\0')
			i++;
	}
	return (nb_words);
}

static char				**split(char *str, char c, char **tab)
{
	int					i;
	int					j;
	int					k;

	i = 0;
	k = 0;
	while (str[i] != '\0')
	{
		while (str[i] == c)
			i++;
		if (str[i] == '\0')
			break ;
		j = 0;
		while (str[i + j] != c && str[i + j] != '\0')
			j++;
		if (!(tab[k] = (char *)malloc(sizeof(char) * (j + 1))))
			return (NULL);
		tab[k] = ft_strncpy(tab[k], str + i, j);
		i += j;
		k++;
	}
	tab[k] = NULL;
	return (tab);
}

char					**ft_strsplit(char *str, char c)
{
	char				**tab;

	if (str == NULL)
		return (NULL);
	if (!(tab = (char **)malloc(sizeof(char *) * (count_words(str, c) + 1))))
		return (NULL);
	tab = split(str, c, tab);
	return (tab);
}
