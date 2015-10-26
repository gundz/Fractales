/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   ft_basename.c                                      :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <fgundlac@student.42.fr>          +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/10/08 20:48:32 by fgundlac          #+#    #+#             */
/*   Updated: 2015/10/08 20:48:39 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

char					*ft_basename(char *path)
{
	int					start;
	int					end;

	end = 0;
	while (path[++end] != '\0')
	{
	}
	end--;
	if (path[0] == '/' && end == 0)
		return (path);
	else if (path[end] == '/' && path[0] == '/')
	{
		path[end] = '\0';
		end--;
	}
	start = end;
	while (path[start] != '/' && start >= 0)
		start--;
	return (path + (start + 1));
}
