from backend.nlp.final_algorithms.sentiment.sentiment_model import SentimentModel

wemb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/word_embeddings/glove.6B.200d.txt"
sent_name_space = "sentiment_model"
sent_tb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/backend/nlp/final_algorithms/sentiment/sent_tb"
aclimdb_path = "/home/berke/Desktop/workspace/bitirme/positive-news/fbackend/aclImdb/"

sent_model = SentimentModel(wembs_path=wemb_path, name_space=sent_name_space, data_path=aclimdb_path,
                            tb_path=sent_tb_path, filter_most_frequent_words=200000)

sent_model.train()
