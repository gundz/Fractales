/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   thread.h                                           :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: fgundlac <marvin@42.fr>                    +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2015/03/16 17:05:18 by fgundlac          #+#    #+#             */
/*   Updated: 2015/04/03 16:07:33 by fgundlac         ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */

#ifndef THREAD_H
# define THREAD_H

# include <pthread.h>

typedef struct		s_tab
{
	int				**tab;
	int				size_x;
	int				size_y;
}					t_tab;

typedef struct		s_thread
{
	t_tab			*tab;
	void			*data;
	int				nb_thread;
	void			*(*f)(void *, int, int);
	int				y_init;
	int				y_max;
	int				x;
	int				y;
}					t_thread;

void				thread_it(t_thread *thread);
t_thread			*get_thread(const int nb_thread, t_tab *const tab,
		void *const data, void *(*f)(void *, int, int));
void				free_thread(t_thread *thread);

t_tab				*get_t_tab(const int x, const int y, const int init);
void				free_t_tab(t_tab *tab);

#endif
