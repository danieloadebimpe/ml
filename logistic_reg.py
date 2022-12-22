import tarfile 
#import pyprind
import pandas as pd
import os, re, nltk
from nltk.stem.porter import PorterStemmer

import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords


df = pd.read_csv("~/terminal-cpu/data/imdb_1k_set.csv", encoding='utf-8') 

# Preprocess the movie dataset into a more convenient format 

print(df.head(3))

print(df.shape)


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


porter = PorterStemmer()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


df['review'] = df['review'].apply(preprocessor)

def tokenizer(text):
    return text.split()



nltk.download('stopwords')


stop = stopwords.words('english')

X_train = df.loc[:25000, 'review'].values 
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values 
y_test = df.loc[25000:, 'sentiment'].values 

print("y_test", y_test.head())


tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer,tokenizer_porter], 
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},

                {'vect__ngram_range': [(1, 1)],
                 'vect__stop_words': [stop, None],
                 'vect__tokenizer': [tokenizer, tokenizer_porter], 
                 'vect__use_idf': [False],
                 'vect__norm': [None],
                 'clf__penalty': ['l1', 'l2'],
                 'clf__C': [1.0, 10.0, 100.0]}
             ]

lr_rfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0, solver='liblinear'))])

gs_lr_tfidf = GridSearchCV(lr_rfidf, param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

