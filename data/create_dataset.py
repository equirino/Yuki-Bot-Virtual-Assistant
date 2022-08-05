import sqlite3
import pandas as pd
import re


date_time = ["2015-05"]


def filter_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    # removing contractions
    sentence = re.sub(r"i'm", "i am", sentence)
    sentence = re.sub(r"he's", "he is", sentence)
    sentence = re.sub(r"she's", "she is", sentence)
    sentence = re.sub(r"it's", "it is", sentence)
    sentence = re.sub(r"that's", "that is", sentence)
    sentence = re.sub(r"what's", "that is", sentence)
    sentence = re.sub(r"where's", "where is", sentence)
    sentence = re.sub(r"how's", "how is", sentence)
    sentence = re.sub(r"\'ll", " will", sentence)
    sentence = re.sub(r"\'ve", " have", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"\'d", " would", sentence)
    sentence = re.sub(r"\'re", " are", sentence)
    sentence = re.sub(r"won't", "will not", sentence)
    sentence = re.sub(r"can't", "cannot", sentence)
    sentence = re.sub(r"n't", " not", sentence)
    sentence = re.sub(r"n'", "ng", sentence)
    sentence = re.sub(r"'bout", "about", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    sentence = re.sub(r"[^a-zA-Z?.!,']+", " ", sentence)
    sentence = sentence.strip()

    return sentence


def main():
    def create_files(file_name, dataframe_index):
        nonlocal dataframe_query
        with open("{}".format(file_name), "a", encoding="utf8") as f:
            for content in dataframe_query["{}".format(dataframe_index)].values:
                if dataframe_index == "parent":
                    processed_content = filter_sentence(content)
                    f.write(processed_content + "\n")

                else:
                    processed_content = filter_sentence(content)
                    f.write(str(processed_content) + "\n")

    for i in date_time:
        sqldb = sqlite3.connect("reddit_data/{}.db".format(i))
        limit = 500
        last_unix = 0
        length = limit
        counter = 0
        test_done = False

        while length == limit:
            dataframe_query = pd.read_sql("SELECT * FROM parent_reply WHERE unix > {} AND parent NOT NULL "
                                          "AND score > 0 ORDER BY UNIX ASC LIMIT {}".format(last_unix, limit), sqldb)
            last_unix = dataframe_query.tail(1)["unix"].values[0]
            length = len(dataframe_query)

            if not test_done:
                create_files("datasets/val_set.from", "parent")
                create_files("datasets/val_set.to", "comment")

                test_done = True
            else:
                create_files("datasets/train_set.from", "parent")
                create_files("datasets/train_set.to", "comment")

            counter += 1
            if counter % 100 == 0:
                print(counter * limit, "rows completed so far")

            if counter * limit % 200000 == 0:
                print("file creation completed")
                break


if __name__ == '__main__':
    main()
