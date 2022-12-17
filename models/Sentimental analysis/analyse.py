# Step 1 - Downloading the data

# nltk.download('stopwords')
# nltk.download('punkt')
import json

import pandas as pd
import io
import re
import string
import random

from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, FreqDist, NaiveBayesClassifier, classify
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


def remove_noise(data_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(data_tokens):

        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|' \
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def convert_token2dict(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def split_tokenize_text(text):
    list_of_lists=[]
    for line in text:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        list_of_lists.append(line_list)
    return list_of_lists


if __name__ == '__main__':
    positive_data = open("D:/MASTER1_SEM1/PROJECTS/Semester_Project/sent_analyse/Data/rt-polarity.pos.txt", "r")
    negative_data = open("D:/MASTER1_SEM1/PROJECTS/Semester_Project/sent_analyse/Data/rt-polarity.neg.txt", "r")
    # df_positive = pd.DataFrame(positive_data, columns=['sentence'])
    # df_negative = pd.DataFrame(negative_data, columns=['sentence'])

    positive_data_tokens = split_tokenize_text(positive_data)
    negative_data_tokens = split_tokenize_text(negative_data)

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    stop_words = stopwords.words('english')

    for tokens in positive_data_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
    for tokens in negative_data_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = convert_token2dict(positive_cleaned_tokens_list)
    negative_tokens_for_model = convert_token2dict(negative_cleaned_tokens_list)

    positive_dataset = [(data_dict, "Positive")
                        for data_dict in positive_tokens_for_model]

    negative_dataset = [(data_dict, "Negative")
                        for data_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset
    random.shuffle(dataset)

    # train_data, test_data = train_test_split(dataset, test_size=0.5)
    train_data = dataset[:8000]
    test_data = dataset[8000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    custom_tweet = "I hate it"

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
