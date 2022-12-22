from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


import numpy as np 
import re , pyprind
from nltk.corpus import stopwords
import time, os
import pickle 



print('training out of core...')
time.sleep(2)

stop = stopwords.words('english')

# preprocessing or "cleaning" function 
# preprocesses one document at a time 
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))

    tokenized = [w for w in text.split() if w not in stop]
    return tokenized



# reads in and returns one document at a time 
# returns the text (review) and the class label (predictor)
def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) 
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

# takes the return value of the stream docs funtion 
# and adds the documents to one list and the class labels to another list
# the purpose of this function is to get the data in a "batch" which is specified by the size parameter
def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


# the countvectorizer requires us to keep vocab in memory 
# the tfidf vectorizer requires us to keep all feature vecs in memory
# with hashingvectorizer, we dont need to store this info in memory 
# making this vectorizer optimial for out-of-core learning 
vect = HashingVectorizer(decode_error='ignore',
                        n_features=2**21,
                        preprocessor=None,
                        tokenizer=tokenizer)

# n_features param sets the number of features (columns) in the resulting 
# output matrix. A small number of features are likely to cause 
# hash collisions, but a large number will increase the number of coefficients (higher dimensionality)

# stochastic gradient descent model 
# an optimization algorithm that updates the model's weights using one example at a time     
# reinitializing the classifier by setting loss to "log_loss"
clf = SGDClassifier(loss='log_loss', random_state=1)
doc_stream = stream_docs(path='movie_data.csv')

# initliaze out-of-core learning:
# progess bar contains 45 iterations.
# as we iterate over 45 minibatches of documents
pbar = pyprind.ProgBar(45)
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()


# we have 50k documents. we just used 45k in lines 76-82
# the remaining 5k we use for testing 
X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)

# we see how our modeln performs on the test (validation) data 
print('Accuracy: %.3f' % clf.score(X_test, y_test))

# we update our classifier 
# the partial_fit function performs one epoch of
# stochastic gradient descent on given sample.
clf = clf.partial_fit(X_test, y_test)

# create directory for the classifier 
# serialize the classifier and stopwords into bytecode
# store them in the classifier directory
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
pickle.dump(stop, 
            open(os.path.join(dest, 'stopwords.pkl'), 'wb'),
            protocol=4)
pickle.dump(clf, 
            open(os.path.join(dest, 'classifier.pkl'), 'wb'),
            protocol=4)

