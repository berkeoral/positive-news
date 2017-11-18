# Codes in here for testing

from threading import Thread

def __start_crawler():
    print("Starting Crawler")
    crawler_thread = Thread(target=Crawler().crawl(), args=())
    crawler_thread.start()

__start_crawler()