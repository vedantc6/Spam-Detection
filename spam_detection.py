
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

df = pd.read_csv("spam.csv", encoding="ISO-8859-1", index_col=False)
# print df.head()

df = df.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"], axis=1)

df.columns = ['labels', 'data']
# print df.head()

# Creating binary labels and putting them into Y list
df['b_labels'] = df['labels'].map({'ham':0, 'spam':1})
Y = df['b_labels'].as_matrix()

# Creating a tf-idf vectorizer or CountVectorizer for data
tfidf = TfidfVectorizer(decode_error="ignore")
X1 = tfidf.fit_transform(df['data'])

count_vectorizer = CountVectorizer(decode_error="ignore")
X2 = count_vectorizer.fit_transform(df['data'])

# Splitting the data into train and test
X_train_tfidf,X_test_tfidf, Y_train_tfidf, Y_test_tfidf = train_test_split(X1, Y, test_size = 0.33)

# Create a model
model_tfidf = MultinomialNB()
model_tfidf.fit(X_train_tfidf,Y_train_tfidf)

print "Train score (TF-IDF): ", model_tfidf.score(X_train_tfidf, Y_train_tfidf)
print "Test score (TF-IDF): ", model_tfidf.score(X_test_tfidf, Y_test_tfidf)

X_train_count,X_test_count, Y_train_count, Y_test_count = train_test_split(X2, Y, test_size = 0.33)

model_count = MultinomialNB()
model_count.fit(X_train_count,Y_train_count)

print "Train score: ", model_count.score(X_train_count, Y_train_count)
print "Test score: ", model_count.score(X_test_count, Y_test_count)

# Visualizing the data
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
# visualize('spam')
# visualize('ham')

# See what is wrong in our model
df['predictions'] = model.predict(X1)

# Things that should be spam but shown as not spam
false_notspam = df[(df['predictions']==0) & (df['b_labels']==1)]['data']
print "Spam messages that have been predicted as not spam:"
for msg in false_notspam:
    print msg
    
    
print "******************************"
print "Non spam messages which have been predicted as spam:"
# Things that should not be spam but reported as spam
false_spam = df[(df['predictions']==1) & (df['b_labels']==0)]['data']
for msg in false_spam:
    print msg



# In[1]:


get_ipython().system(u'pip install wordcloud')

