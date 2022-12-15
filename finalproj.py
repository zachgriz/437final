from collections import Counter
from sklearn.feature_extraction.text import *
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from nltk.corpus import stopwords
import random
import pandas as pd
import numpy as np
import re

class MulticlassPerceptron:
  def __init__(self, num_classes, num_features):
    # initialize the weights to 0
    self.weights = np.zeros((num_classes, num_features))
    
  def predict(self, x, test = False):
    # compute the dot product of the input features and the weights
    dp = x.dot(self.weights.T)
    # make a prediction 
    return np.argmax(dp, axis=1) if test is True else np.argmax(dp)
    
  def fit(self, x, y, num_epochs):
    # train the model for a given number of epochs
    for epoch in range(num_epochs):
      # loop through the training examples
      for x_i, y_i in zip(x, y):
        # make a prediction on the training example
        prediction = self.predict(x_i)
        # if the prediction is not correct, update the weights
        if prediction != y_i:
          self.weights[prediction, :] -= x_i
          self.weights[y_i, :] += x_i


headers=['Tweet_ID','Entity','Sentiment','Tweet_content']
# read data from training dataset
training=pd.read_csv('archive/twitter_training.csv', sep=',', names=headers)
# read data from validation dataset
testing=pd.read_csv('archive/twitter_validation.csv', sep=',', names=headers)

# drop duplicate entries
training.drop_duplicates(inplace=True)
testing.drop_duplicates(inplace=True)

#drop null values
training.dropna(axis=0, inplace=True)
testing.dropna(axis=0, inplace=True)

# reset indices
training.reset_index(inplace=True)
testing.reset_index(inplace=True)

# convert target values to numeric values
lb = preprocessing.LabelEncoder()
training['Sentiment']=lb.fit_transform(training['Sentiment'])
testing['Sentiment']=lb.fit_transform(testing['Sentiment'])



X_train = training["Tweet_content"]
X_test = testing["Tweet_content"]

y_train = training["Sentiment"]
y_test = testing["Sentiment"]

REPLACE_WITH_SPACE = re.compile("(@)")
SPACE = " "

# convert lines to lowercase and remove '@' signs
X_train_clean = [REPLACE_WITH_SPACE.sub(SPACE, line.lower()) for line in X_train]
X_test_clean = [REPLACE_WITH_SPACE.sub(SPACE, line.lower()) for line in X_test]

# create vectorizer
#remove english stop words from the data points
vectorizer = CountVectorizer(stop_words=stopwords.words('english'), binary=True)

# convert the data to a matrix of token counts
Data = vectorizer.fit_transform(X_train_clean + X_test_clean)

train_data = Data[:len(X_train_clean)]
test_data = Data[len(X_train_clean):len(X_train_clean)+len(X_test_clean)]

# Logistic Regression
lr = LogisticRegression(max_iter=500).fit(train_data, y_train)

lr_pred = lr.predict(test_data)

print('Logistic Regression Accuracy: %.2f' % accuracy_score(y_test, lr_pred))

# # My Perceptron

ppn = MulticlassPerceptron(4, Data.shape[1])

ppn.fit(train_data, y_train, 5)
ppn_pred = ppn.predict(test_data, test=True)

print('My Perceptron Accuracy: %.2f' % accuracy_score(y_test, ppn_pred))

# SKLearn Perceptron

ppn = Perceptron(max_iter=50).fit(train_data, y_train)

ppn_pred = ppn.predict(test_data)

print('SKlearn Perceptron Accuracy: %.2f' % accuracy_score(y_test, ppn_pred))

# # Naive Bayes

nbc = MultinomialNB().fit(train_data, y_train)

nbc_pred = nbc.predict(test_data)

print('Naive Bayes Accuracy: %.2f' % accuracy_score(y_test, nbc_pred))


