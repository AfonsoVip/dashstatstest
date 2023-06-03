import os
import time
import shutil

class Session:
    def add_result(self, conn, file_name, timestamp, threshold):
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO history (file_name, timestamp, threshold) VALUES (?, ?, ?)''', (file_name, timestamp, threshold))
        conn.commit()

    def get_history(self, conn):
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history")
        records = cursor.fetchall()
        history = [{'id': row[0], 'file_name': row[1], 'timestamp': row[2], 'threshold': row[3]} for row in records]
        return history