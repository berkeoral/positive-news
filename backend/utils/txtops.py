import os
import filelock
import glob

class TextOps:
    def __init__(self):
        self.filename = "nlpdb.txt"
        self.test_filename= "test.txt"
        self.group_sep = 29
        self.record_sep = 30


    def append_records(self, url, title, text):
        with filelock.FileLock(self.filename):
            if os.path.exists(self.filename):
                mode = 'a'
            else:
                mode = 'w'
            file = open(self.filename, mode, encoding='utf-8')
            file.write(url + chr(self.group_sep))
            file.write(title + chr(self.group_sep))
            file.write(text + chr(self.record_sep))

    def append_test(self, url, title, text, sent_class):
        with filelock.FileLock(self.test_filename):
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
        with filelock.FileLock(filename):
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                temp_list = content.split(chr(self.record_sep))
                for i in range(len(temp_list)):
                    ret_list.append(temp_list[i].split(chr(self.group_sep)))
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
            if( int(sent_lvl) <= 10 and int(sent_lvl) >= -10 ):
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
        return acmimdb














