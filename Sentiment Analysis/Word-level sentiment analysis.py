### Read and save the words and thier labels#####
import numpy as np
import io
import re
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize
from pyarabic.araby import tokenize, is_arabicrange, strip_tashkeel
import joblib
import pyarabic.araby as araby
import pandas as pd
positive_data = pd.read_csv('positive lexicon.csv' ,encoding = "utf-8")
positive= positive_data["data"].values.tolist()
negative_data = pd.read_csv('negative lexicon.csv' ,encoding = "utf-8")
negative= negative_data["data"].values.tolist()


text = []
labels = []

negative_labels = []
for x in range(0,len(negative)):
    negative_labels.append(0)
  
positive_labels=[]
for x in range(0,len(positive)):
    positive_labels.append(1)


## Store all the words in one list ######
texts = negative + positive
labels = negative_labels + positive_labels















#### Load the model ##############
model = Word2Vec.load(model_name)
    #model = FastText.load_fasttext_format(model_name)
def word_embedding(text, label):
        embeddings = []
        if text in model.wv.vocab:
            embeddings = model.wv[text]
            return embeddings, label
        else:
            # Handle case when no embeddings are found
            return None, None
    # Split dataset into train and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)
    # Convert training texts to average word embeddings
train_embeddings = []
trained_labels = []
index = 0
for text in train_texts:
        embedding, final_train_label = word_embedding(text, train_labels[index])
        if embedding is not None:
            train_embeddings.append(embedding)
            trained_labels.append(final_train_label)
        index += 1
    #train_embeddings = np.array(train_embeddings).reshape(-1, 1)
    # Train SVM classifier
svm_classifier = svm.SVC()
svm_classifier.fit(train_embeddings, trained_labels)
joblib.dump(svm_classifier, 'svm_classifier_model.pkl')
    # Convert test texts to average word embeddings
test_embeddings = []
tested_labels = []
index = 0
for text in test_texts:
        embedding, final_test_label = word_embedding(text, test_labels[index])
        if embedding is not None:
            test_embeddings.append(embedding)
            tested_labels.append(final_test_label)
        index += 1
    # Predict using the trained SVM classifier
predictions = svm_classifier.predict(test_embeddings)
    # Calculate accuracy
accuracy = accuracy_score(tested_labels, predictions)
print("Accuracy:", accuracy)
