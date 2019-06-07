import pandas as pd
from data_preprocessor.get_user_data import get_access_token, get_refresh_token, instantiate_user
from application_secrets.fitbit_application_secrets import *


class UserTokenGenerator:
    def __init__(self, token_file_path=f'../../data/user_tokens/tokens.csv'):
        self.__token_file_path = token_file_path

    def get_token_file_path(self):
        return self.__token_file_path

    def set_token_file_path(self, new_token_file_path):
        self.__token_file_path = new_token_file_path

    def get_tokens_using_user_names(self, user_name: str):
        token_information = pd.read_csv(self.get_token_file_path(), index_col='client_id')
        if user_name.lower() in token_information['user_name'].values:
            return token_information['access_token'][token_information['user_name'] == user_name.lower()].iloc[0], \
                   token_information['refresh_token'][token_information['user_name'] == user_name.lower()].iloc[0], \
                   token_information[token_information['user_name'] == user_name.lower()].index[0], \
                   token_information['client_secret'][token_information['user_name'] == user_name.lower()].iloc[0]
        else:
            raise ValueError(f'Invalid User name')

    def add_user_information(self, user_name):
        _, _, server = instantiate_user(user_name)
        ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
        CLIENT_ID, CLIENT_SECRET = get_fitbit_client_id(user_name), get_fitbit_client_secret(user_name)
        token_information.loc[CLIENT_ID] = \
            {'user_name': user_name.lower(), 'client_secret': CLIENT_SECRET,
             'access_token': ACCESS_TOKEN, 'refresh_token': REFRESH_TOKEN}
        token_information.to_csv(self.get_token_file_path())
        return ACCESS_TOKEN, REFRESH_TOKEN, CLIENT_ID, CLIENT_SECRET

    # def get_tokens_using_user_id(self, user_id: str):
    #     token_information = pd.read_csv(self.get_token_file_path())
    #     if user_id.lower() in token_information['user_id'].values:
    #         return token_information['access_token'][token_information['user_id'] == user_id].iloc[0], \
    #                token_information['refresh_token'][token_information['user_id'] == user_id].iloc[0]
    #     else:
    #
    #         _, _, server = instantiate_user(user_name)
    #         return get_access_token(server), get_refresh_token(server)
