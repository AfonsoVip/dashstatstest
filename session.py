import os
import time
import shutil

class Session:
    def __init__(self):
        self.history = []

    def add_result(self, file_name, timestamp, result):
        entry = {'file_name': file_name, 'timestamp': timestamp, 'result': result}
        self.history.append(entry)