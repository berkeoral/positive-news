import csv
import os
import sys

import filelock
import glob

from tqdm import tqdm


class TextOps:
    def __init__(self):
        self.filename = "nlpdb.txt"
        self.test_filename = "test.txt"
        self.group_sep = 29
        self.record_sep = 30

    def append_records(self, url, title, text):
        if os.path.exists(self.filename):
            mode = 'a'
        else:
            mode = 'w'
        file = open(self.filename, mode, encoding='utf-8')
        file.write(url + chr(self.group_sep))
        file.write(title + chr(self.group_sep))
        file.write(text + chr(self.record_sep))

    def append_records_v2(self, filename, n_record):
        if os.path.exists(filename):
            mode = 'a'
        else:
            mode = 'w'
        file = open(filename, mode, encoding='utf-8')
        for i in range(len(n_record)):
            if i != len(n_record) - 1:
                file.write(n_record[i] + chr(self.group_sep))
            else:
                file.write(n_record[i] + chr(self.record_sep))

    def append_test(self, url, title, text, sent_class):
        if os.path.exists(self.test_filename):
            mode = 'a'
        else:
            mode = 'w'
        file = open(self.test_filename, mode, encoding='utf-8')
        file.write(url + chr(self.group_sep))
        file.write(title + chr(self.group_sep))
        file.write(sent_class + chr(self.group_sep))
        file.write(text + chr(self.record_sep))

    def records_as_list(self, filename):
        temp_list = []
        ret_list = []
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            temp_list = content.split(chr(self.record_sep))
            for i in range(len(temp_list)):
                ret_list.append(temp_list[i].split(chr(self.group_sep)))
        del ret_list[-1]  # last element always empty
        return ret_list

    def tag_papers_(self):
        articles = self.records_as_list(self.filename)
        test_size = len(self.records_as_list(self.test_filename))
        article_length = len(articles)
        while test_size < article_length:
            print(articles[test_size][1])
            print(articles[test_size][2])
            sent_lvl = input("-10 / 10: ")
            if sent_lvl == "e":
                return
            if (int(sent_lvl) <= 10 and int(sent_lvl) >= -10):
                self.append_test(articles[test_size][0], articles[test_size][1], articles[test_size][2], sent_lvl)
                test_size += 1
            print(" ")
            print("----------------------------------------------------------------------------------------")
            print(" ")

    # Returns training and testing together
    # 0 is id, 1 is score, 2 is text
    def acmimdb_as_list(self, source_path):
        folders = ["train/pos/*.txt", "train/neg/*.txt", "test/pos/*.txt", "test/neg/*.txt"]
        acmimdb = []
        pbar = tqdm(50000, file=sys.stdout)
        for folder in folders:
            for file_path in glob.glob(source_path + folder):
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        filename = os.path.basename(file_path)
                        details = filename.split('_')
                        acmimdb.append([details[0], details[1].split('.')[0], content])
                except IOError:
                    pass
                finally:
                    pbar.update(1)
        return acmimdb

    # 0 is original text, 1 is summarized text
    @staticmethod
    def indian_news_summary_as_list(source_path):
        ret = []
        with open(source_path, encoding="ISO-8859-1") as file:
            readCSV = csv.reader(file, delimiter=',')
            for row in readCSV:
                ret.append([row[2], row[3], row[5], row[4]])
        ret.pop(0)
        return ret

    @staticmethod
    def cnn_dailymail_as_list(base_path):
        ret = []
        for ds in os.listdir(base_path):
            _ds_path = base_path + "/%s" % ds
            for file_path in tqdm(os.listdir(_ds_path), file=sys.stdout):
                with open(_ds_path + "/%s" % file_path, encoding="utf-8") as file:
                    content = file.read()
                    summ_ind = content.index("@highlight")
                    article = content[:summ_ind]
                    highlights = content[summ_ind:].split("@highlight")
                    highlights = [highlight for highlight in highlights if highlight != ""]
                    ret.append([article, highlights])
        return ret
