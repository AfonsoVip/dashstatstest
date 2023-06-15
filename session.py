import os
import sqlite3
import time
import shutil
import json

class Session:
    def __init__(self, db_path="results.db"):
        self.conn = sqlite3.connect(db_path)
        self._create_table()

    def _create_table(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY,
                file_name TEXT,
                timestamp TEXT,
                threshold REAL,
                result TEXT
            )
        """)
        self.conn.commit()

    def add_result(self, file_name, timestamp, threshold, result):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO history (file_name, timestamp, threshold, result) VALUES (?, ?, ?, ?)''', (file_name, timestamp, threshold, json.dumps(result)))
        self.conn.commit()

    def get_history(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM history")
        records = cursor.fetchall()
        history = [{'id': row[0], 'file_name': row[1], 'timestamp': row[2], 'threshold': row[3], 'result': json.loads(row[4])} for row in records]
        return history
