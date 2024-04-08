import psycopg2
import pandas as pd

class DatabaseConnector:
    def __init__(self, args):
        self.args = args

    def send_sql_query(self,query: str):
        """
        Выполняет запрос к базе.

        :param query: строка с sql запросом.
        :param args: аргументы для подключения в БД.
        """
        conn = psycopg2.connect(**self.args)
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
        except (Exception, psycopg2.Error) as error:
            print("Error while fetching data from PostgreSQL", error)
        finally:
            if conn:
                cursor.close()
                conn.close()

    def get_df_from_query(self,query: str,):
        """
        Выполняет запрос к базе.

        :param query: строка с sql запросом.
        :param args: аргументы для подключения в БД.

        :return df: датафрейм с результатом.
        """
        conn = psycopg2.connect(**self.args)
        df = pd.read_sql(query, conn)
        conn.close()
        return df
