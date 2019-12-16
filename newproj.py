# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 05:50:02 2019

@author: Dell
"""
#import numpy as np
import pandas as pd
#importing the dataset
dataset= pd.read_csv('Movie_Reviews.tsv',delimiter='\t',quoting=3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    #Creating the bag of WOrds model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features = 1500)
X= cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values
#Splitting  the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.20, random_state=0)

#Fitting Naive Bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting test results
y_pred= classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)

Xi=[[0,0,0,1,1,1][1,1,1,0,0,0]]
print(classifier.predict(Xi))

