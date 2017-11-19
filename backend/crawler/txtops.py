import os
from pip._vendor.lockfile import FileLock


class TextOps:
    def __init__(self):
        self.filename = "nlpdb.txt"
        self.group_sep = 35
        self.record_sep = 36

    def append_file(self, url, title, text):
        with FileLock(self.filename):
            if os.path.exists(self.filename):
                mode = 'a'
            else:
                mode = 'w'
            file = open(self.filename, mode, encoding='utf-8')
            file.write(url + chr(self.group_sep))
            file.write(title + chr(self.group_sep))
            file.write(text + chr(self.record_sep))

    def records_as_list(self):
        temp_list = []
        ret_list = []
        with FileLock(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as file:
                content = file.read()
                temp_list = content.split(chr(self.record_sep))
                for i in range(len(temp_list)):
                    ret_list.append(temp_list[i].split(chr(self.group_sep)))
        return ret_list








