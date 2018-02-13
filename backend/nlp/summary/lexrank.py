from backend.utils.txtops import TextOps


class LexRank:
    def __init__(self, data_dir, debug=10):
        self.text_ops = TextOps()
        self.raw_data = self.text_ops.records_as_list(data_dir)
        self.idf = {}
        self.__calc_idf()

    def __calc_idf(self):
        return None