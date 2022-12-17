# Step 1 - Downloading the data

import nltk
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('stopwords')
#nltk.download('punkt')

import collections

from nltk import (precision, recall, f_measure)
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import json

import pandas as pd
import io
import re
import string
import random
import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

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
    positive_data = open("C:/Users/Colmr/Desktop/TrainedModel/rt-polarity.pos.txt", "r")
    negative_data = open("C:/Users/Colmr/Desktop/TrainedModel/rt-polarity.neg.txt", "r")
#    df_positive = pd.DataFrame(positive_data, columns=['sentence'])
 #   df_negative = pd.DataFrame(negative_data, columns=['sentence'])

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

  #  train_data, test_data = train_test_split(dataset, test_size=0.5)
    train_data = dataset[:8000]
    test_data = dataset[8000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))
    

    print(classifier.show_most_informative_features(10))
    

    custom_tweet = "I love watching movies, it's my favourite pasttime"

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
    labels = [] # Pos / Neg
    for token in custom_tokens:
        labels.append(token)
        
    #list_of_sentences = ["I love watching movies, it's my favourite pasttime", 'I enjoy watching the sun set', "The dog bit me", "I lost my eyesight in 2014 in a car accident."]
     #   
    #def testing_list_of_sentences(list): #custom token needs to be changed
    #    custom_tokens = remove_noise(word_tokenize(custom_tweet))
    #    for i in range( 0, len(list)):
    #        print(list[i], classifier.classify(dict([token, True] for token in custom_tokens)))
    #        labels = [] # Pos / Neg
    #        for token in custom_tokens:
    #            labels.append(token)

    #testing_list_of_sentences(list_of_sentences)


    ########   Evaluates Model ############### 

    def true_evaluate_model(test_data):
        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)
 
        for i, (feats, label) in enumerate(test_data):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)
 
        print( 'Precision:', nltk.precision(refsets['Positive'], testsets['Positive']) )
        print( 'Recall:', nltk.recall(refsets['Positive'], testsets['Positive']) )
        print ('pos precision:', nltk.precision(refsets['Positive'], testsets['Positive']))
        print ('pos recall:', nltk.recall(refsets['Positive'], testsets['Positive']))
        print ('pos F-measure:', nltk.f_measure(refsets['Positive'], testsets['Positive']))
        print ('neg precision:', nltk.precision(refsets['Negative'], testsets['Negative']))
        print ('neg recall:', nltk.recall(refsets['Negative'], testsets['Negative']))
        print ('neg F-measure:', nltk.f_measure(refsets['Negative'], testsets['Negative']))

    true_evaluate_model(test_data)
    # comment out
   # def evaluate_model(test_data) -> dict:
    #    dataset, labels = zip(*test_data)
        #dataset => dictionary of words & booleans (TRUE)
        #labels => Positive / Negative

     #   true_positives = 0
       # false_positives = 1e-8  # Can't be 0 because of presence in denominator
      #  true_negatives = 0
        #false_negatives = 1e-8

#        for i, dataset in enumerate(dataset):
 #           true_label = labels[i]
  #          for predicted_label, score in dataset.items(): 
   #             if (predicted_label == "Negative"):
    #                continue
     #           if score >= 0.5 and true_label == "Positive":
      #              true_positives += 1
       #         elif score >= 0.5 and true_label == "Negative":
        #            false_positives += 1
         #       elif score < 0.5 and true_label == "Negative":
           #         true_negatives += 1
          #      elif score < 0.5 and true_label == "Positive":
            #        false_negatives += 1

#        print("TP - {} \t FP - {} \t TN {} \t FN {} \t".format(true_positives, false_positives, true_negatives, false_negatives))
 #       
   #     precision = true_positives / (true_positives + false_positives)
  #      recall = true_positives / (true_positives + false_negatives)

    #    if precision + recall == 0:
     #       f_score = 0
      #  else:
       #     f_score = 2 * (precision * recall) / (precision + recall)
        #return {"precision": precision, "recall": recall, "f-score": f_score}
            
#    evaluate_model(test_data)
    #evaluate_model(train_data)