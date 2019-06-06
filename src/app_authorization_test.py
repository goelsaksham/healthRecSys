from data_preprocessor.get_user_data import *


def main():
    CLIENT_ID = '22D8XW'
    CLIENT_SECRET = '39823c504eeb610fed6be977d3806cee'
    # server = Oauth2.OAuth2Server(CLIENT_ID, CLIENT_SECRET)
    # server.browser_authorize()
    # ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(server), get_refresh_token(server)
    # print(f'ACCESS_TOKEN: {ACCESS_TOKEN}\nREFRESH_TOKEN: {REFRESH_TOKEN}')
    ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIyMkQ4WFciLCJzdWIiOiI2V1FSRjUiLCJpc3MiOiJGaXRiaXQiLCJ0eXAiOiJhY2Nlc3NfdG9rZW4iLCJzY29wZXMiOiJyc29jIHJzZXQgcmFjdCBybG9jIHJ3ZWkgcmhyIHJwcm8gcm51dCByc2xlIiwiZXhwIjoxNTU5ODQ5MjUyLCJpYXQiOjE1NTk4MjA0NTJ9.zey1bB58ndvmiKQx4n8ZEZtfqEQjqU8GcUkEbnNBG0M'
    REFRESH_TOKEN = '338f4991eecd03cd7de7aad036fda118af8fb8f8351a97a814e98d05f73bc82d'
    auth_client = get_auth_client(CLIENT_ID, CLIENT_SECRET, ACCESS_TOKEN, REFRESH_TOKEN)
    print(get_heart_intraday(get_heart_data(auth_client, '2019-02-27'), '6WQRF5'))


if __name__ == '__main__':
    main()
