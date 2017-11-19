import time
import threading

from backend.crawler.crawler import Crawler
from backend.crawler.txtops import TextOps

CRAWLER_CALL_INTERVAL = 5
CRAWLER_INIT_PADDING = 5
CRAWLER_THREAD_NAME = "non_deamon"


def __start_crawler():
    print("Starting crawler")
    arrl = TextOps().records_as_list()
    crawler_thread = threading.Thread(target=Crawler().crawl)
    crawler_thread.start()
    crawler_thread.join()
    __start_crawler()

def main():
    __start_crawler()


if __name__ == "__main__":
    main()
