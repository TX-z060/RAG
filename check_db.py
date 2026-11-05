from db import get_connection

with get_connection() as connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT DATABASE() AS database_name, @@port AS port")
        result = cursor.fetchone()
        print(result)