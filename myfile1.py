#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis Using Naive Bayes

# We will perform sentiment analysis on the IMDB dataset, which has 25k positive and 25k negative movie reviews. We built an NB classifier that classifies an unseen movie review as positive or negative. Sentiment analysis, also known as opinion mining, is a powerful natural language processing (NLP) technique that aims to determine the sentiment or emotional tone expressed within a piece of text, such as a tweet, a product review, or a news article. This invaluable tool enables us to automatically categorize and analyze text data, identifying whether the expressed sentiment is positive or negative.

# # Importing necessary packages for text manipulation and loading the dataset into a pandas dataframe

# In[13]:


import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('omw-1.4')
data = pd.read_csv('../Desktop/NaiveBayesCOA/data/IMDB Dataset.csv')
data


# # Data Preprocessing

# We remove HTML tags, URLs and non-alphanumeric characters from the dataset using regex functions. Stopwords (commonly used words like ‘and’, ‘the’, ‘at’ that do not hold any special meaning in a sentence) are also removed from the corpus using the nltk stopwords list 

# In[2]:


def remove_tags(string):
    # Remove HTML tags
    result = re.sub(r'<[^>]+>', ' ', string)
    # Remove URLs
    result = re.sub(r'https?://\S+?', '', result)
    # Remove non-alphanumeric characters and convert to lowercase
    result = re.sub(r'[^a-zA-Z0-9]', ' ', result)
    result = result.lower()
    return result
data['review']=data['review'].apply(lambda cw : remove_tags(cw)) 
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].str.split().apply(lambda x: ' '.join([word for word in x if word not in stop_words]))


# On performing lemmatization on the text(Lemmatization is used to find the root form of words or lemmas in NLP. For example, the lemma of the words reading, reads, read is read) This helps save unnecessary computational overhead in trying to decipher entire words since the meanings of most words are well-expressed by their separate lemmas. We perform lemmatization using the WordNetLemmatizer() from nltk. The text is first broken into individual tokens/ words using the WhitespaceTokenizer() from nltk. We write a function lemmatize_text to perform lemmatization on the individual words.

# In[3]:


w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)
data['review'] = data.review.apply(lemmatize_text)
print(data)


# # Encoding Labels and Making Train-Test Splits
# LabelEncoder() from sklearn.preprocessing is used to convert the labels (‘positive’, ‘negative’) into 1’s and 0’s respectively.

# In[4]:


reviews = data['review'].values
labels = data['sentiment'].values
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)


# The dataset is then split into 80% train and 20% test parts using train_test_split from sklearn.model_selection.

# In[5]:


train_sentences, test_sentences, train_labels, test_labels = train_test_split(reviews, encoded_labels, stratify = encoded_labels)


# # Building the Naive Bayes Classifier
# Many variants of the Naive Bayes classifier are available in the sklearn library. However, we are going to be building our own classifier from scratch using the formulas described earlier. We start by using the CountVectorizer from sklearn.feature_extraction.text to get the frequency of each word appearing in the training set. We store them in a dictionary called ‘word_counts’. All the unique words in the corpus are stored in ‘vocab’.

# In[6]:


vec = CountVectorizer(max_features=3000)
X = vec.fit_transform(train_sentences)
vocab = vec.get_feature_names_out()
X = X.toarray()
word_counts = {}
for l in range(2):
    word_counts[l] = defaultdict(lambda: 0)
for i in range(X.shape[0]):
    l = train_labels[i]
    for j in range(len(vocab)):
        word_counts[l][vocab[j]] += X[i][j]


# we need to perform Laplace smoothing(adding a small positive value to each of the existing conditional probability values to avoid zero values in the probability model) to handle words in the test set which are absent in the training set. We define a function ‘laplace_smoothing’ which takes the vocabulary and the raw ‘word_counts’ dictionary and returns the smoothened conditional probabilities.

# In[7]:


def laplace_smoothing(n_label_items, vocab, word_counts, word, text_label):
    a = word_counts[text_label][word] + 1
    b = n_label_items[text_label] + len(vocab)
    return math.log(a/b)


# Defining Fit and Predict functions for our classifier

# In[8]:


def group_by_label(x, y, labels):
    data = {}
    for l in labels:
        data[l] = x[np.where(y == l)]
    return data
def fit(x, y, labels):
    n_label_items = {}
    log_label_priors = {}
    n = len(x)
    grouped_data = group_by_label(x, y, labels)
    for l, data in grouped_data.items():
        n_label_items[l] = len(data)
        log_label_priors[l] = math.log(n_label_items[l] / n)
    return n_label_items, log_label_priors


# In[10]:


def predict(n_label_items, vocab, word_counts, log_label_priors, labels, x):
    result = []
    for text in x:
        label_scores = {l: log_label_priors[l] for l in labels}
        words = set(w_tokenizer.tokenize(text))
        for word in words:
            if word not in vocab : 
                continue
            for l in labels:
                log_w_given_l = laplace_smoothing(n_label_items, vocab, word_counts, word, l)
                label_scores[l] += log_w_given_l
        result.append(max(label_scores, key=label_scores.get))
    return result


# The ‘fit’ function takes x (reviews) and y (labels – ‘positive’, ‘negative’) values to be fitted on and returns the number of reviews with each label and the apriori conditional probabilities. Finally, the ‘predict’ function is written which returns predictions on unseen test reviews.

# In[12]:


labels = [0,1]
n_label_items, log_label_priors = fit(train_sentences,train_labels,labels)
pred = predict(n_label_items, vocab, word_counts, log_label_priors, labels, test_sentences)
print("Accuracy of prediction on test set : ", accuracy_score(test_labels,pred))


# In[ ]:



