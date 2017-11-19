from threading import Thread

import time

from backend.crawler.crawler import Crawler

CRAWLER_CALL_SPACE = 60


def __start_crawler():
    crawler_thread = Thread(target=Crawler().crawl(), args=())
    while True:
        if crawler_thread.is_alive():
            print("Starting Crawler")
            crawler_thread.start()
        time.sleep(CRAWLER_CALL_SPACE)


def main():
    __start_crawler()


if __name__ == "__main__":
    main()
