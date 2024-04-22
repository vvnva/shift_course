import os

DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DATA_FULL_PATH ="/Users/vi/home-credit-default-risk/"
DATABASE_NAME = 'postgres'

DB_ARGS = {
    'database': DATABASE_NAME,
    'user': DB_USER,
    'password': DB_PASSWORD,
    'host': '127.0.0.1',
    'port': '5432',
}
