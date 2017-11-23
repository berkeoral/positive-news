import os

import newspaper
from pip._vendor.lockfile import FileLock


class Sources:
    def __init__(self):
        self.filename = "sources.txt"

    def get_sources(self):
        ret_list = []
        with FileLock(self.filename):
            with open(self.filename, 'r', encoding='utf-8') as file:
                content = file.read()
                temp_list = content.split(" ")
                for i in range(len(temp_list)):
                    ret_list.append(temp_list[i])
        return ret_list[0:len(ret_list)-1]

    @staticmethod
    def get_popular_urls():
        return newspaper.popular_urls()

    def add_source(self, url):
        with FileLock(self.filename):
            if os.path.exists(self.filename):
                mode = 'a'
            else:
                mode = 'w'
            file = open(self.filename, mode, encoding='utf-8')
            file.write(url + " ")

    def refresh_popular_sources(self):
        for source in newspaper.popular_urls():
            self.add_source(source)
