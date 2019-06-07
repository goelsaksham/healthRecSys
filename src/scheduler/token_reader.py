import pandas as pd
import datetime
from data_preprocessor.get_user_data import get_access_token, get_refresh_token, instantiate_server, get_expires_at, \
    get_user_information, get_fitbit_user_id


class UserTokenGenerator:
    def __init__(self, token_file_path=f'../../data/user_tokens/tokens.csv'):
        self.__token_file_path = token_file_path

    def get_token_file_path(self):
        return self.__token_file_path

    def set_token_file_path(self, new_token_file_path):
        self.__token_file_path = new_token_file_path

    def token_hasnt_expired(self, expire_time):
        current = datetime.datetime.now()
        expire = datetime.datetime.fromtimestamp(expire_time)
        return current < expire

    def get_tokens_using_user_names(self, user_name: str):
        token_information = pd.read_csv(self.get_token_file_path(), index_col='user_id')
        if user_name.lower() in token_information['user_name'].values:
            expire_date = token_information['expires_at'][token_information['user_name'] == user_name.lower()].iloc[0]
            if self.token_hasnt_expired(expire_date):
                return token_information[token_information['user_name'] == user_name.lower()].index[0],\
                       token_information['access_token'][token_information['user_name'] == user_name.lower()].iloc[0], \
                       token_information['refresh_token'][token_information['user_name'] == user_name.lower()].iloc[0]
            else:
                return self.add_user_information(user_name)
        else:
            return self.add_user_information(user_name)

    def add_user_information(self, user_name):
        _, _, server = instantiate_server()
        ACCESS_TOKEN, REFRESH_TOKEN, EXPIRES_AT = get_access_token(server), get_refresh_token(server), \
                                                  get_expires_at(server)
        user_id = get_fitbit_user_id(get_user_information(server))
        with open(self.get_token_file_path(), 'a') as file_writer:
            file_writer.write(f'\n{user_id},{user_name.lower()},{ACCESS_TOKEN},{REFRESH_TOKEN},{EXPIRES_AT}')
        return user_id, ACCESS_TOKEN, REFRESH_TOKEN
