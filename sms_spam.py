# importing the libraries 

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# loading the dataset

df = pd.read_csv('sms_spam.csv', encoding='ISO-8859-1')

# dropping the unnecessary columns

df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# rename the columns to something else

df.columns = ['labels', 'data']

# creating a binary mapping 
df['b_labels']=  df['labels'].map({'spam':1, 'ham':0})
Y = df['b_labels'].values 
 
# converting the dataset from text to feature vectors using the TF-IDF algorithm

vectorizer = TfidfVectorizer(decode_error='ignore')

X = vectorizer.fit_transform(df['data'])
print(X)

X_train, X_test, y_train, y_test  = train_test_split(X, Y, test_size=0.33)

# training the model

model1 = MultinomialNB()
model1.fit(X_train, y_train)
score1 = model1.score(X_test, y_test)

model2 = AdaBoostClassifier()
model2.fit(X_train, y_train)
score2 = model2.score( X_test, y_test)

if(score1>score2):
	predictions = model1.predict(X_test)
else:
	predictions = model2.predict(X_test)

correct = 0
for i in range(0, len(y_test)):
	if predictions[i] == y_test[i]:
		correct = correct+1
correct_percent = correct/len(predictions)
print("accuracy of the predictions is:", correct_percent)

# testing on taking live input
test = input("enter the text: ")
test1 = ["Python's popularity has propelled once again, with a major driver of its growth being Data Science—one of the fastest growing global business trends. Like yin and yang, Data Science and Python are now two inseparable forces that stand mightily on their own, but together, they produce an even more powerful energy."]
trial = vectorizer.fit_transform(test1)
predictions1 = model2.predict(trial)
print(predictions1)