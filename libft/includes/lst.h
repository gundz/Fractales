/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   lst.h                                              :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/18 09:49:38 by fgundlac          #+#    #+#             */
/*   Updated: 2015/03/18 10:02:57 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef LST_H
# define LST_H

typedef struct		s_list
{
	struct s_list	*next;
	void			*data;
}					t_list;

t_list				*lst_create_elem(void *data);
void				lst_push_back(t_list **lst, void *data);
void				lst_b_sort(t_list **lst, int (*cmp)(), const int rev);
void				lst_free(t_list **lst, const int free_data);
char				*lst_to_char(t_list *lst);
int					lst_csize(t_list *lst);
unsigned int		lst_count(t_list *lst);

#endif
