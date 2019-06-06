"""
This file contains the mapping for user_names to their user_ids that we will use for iterating through all the users
to collect the data.
"""

all_users_info = {
	'Saksham': {
		'user_id': '6WQRF5',
		},
	'Abhiraj': {
		'user_id': '6XV49J',
		},
	'Meghna': {
		'user_id': '4WT27W',
		},
	'Prof': {
		'user_id': '6WQRF5',
		}
	}


def get_user_ids(user_name):
	return all_users_info[user_name]['user_id']