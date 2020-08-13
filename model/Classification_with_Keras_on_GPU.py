import pandas as pd
import numpy as np
import sklearn
import gensim
from nltk import tokenize
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from scipy import spatial
from sklearn import model_selection, naive_bayes, svm
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import *
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import os

VALIDATION_SPLIT = 0.2


def read_data():
    data = pd.read_csv("model_training.csv", header = 0)
    #df = data.head(1000) 
    df = data
    return df


def word2VecModel(data):
    w2v = gensim.models.Word2Vec(list(data["parsed_description"]), size=100, window=10, min_count=1, iter=20)
    docs_vectors = pd.DataFrame() # creating empty final dataframe
    for doc in data['parsed_description']: # looping through each document and cleaning it
        #print(doc)
        for each_word in doc:
            # creating a temporary dataframe(store value for 1st doc & for 2nd doc remove the details of 1st & proced through 2nd and so on..)
            temp = pd.DataFrame()  
            try:
                word_vec = w2v[each_word]
                #print(word_vec)
                temp = temp.append(pd.Series(word_vec), ignore_index = True) # if word is present then append it to temporary dataframe
                #print(temp)
            except:
                #print("inside except")
                pass
            doc_vector = temp.mean() # take the average of each column(w0, w1, w2,........w300)
        docs_vectors = docs_vectors.append(doc_vector, ignore_index = True) # append each document value to the final dataframe
    return docs_vectors
    
def encodeLabel(doc_vector):
    lb = LabelEncoder()
    y = lb.fit_transform(doc_vector['category'])
    y_df = pd.DataFrame(y)
    doc_vector['category'] = y_df
    return doc_vector


def makeTfModel(data):
    max_length  = max([len(s) for s in data['parsed_description']])
    tokenizer_obj = Tokenizer()
    EMBEDDING_DIM = 100
    tokenizer_obj.fit_on_texts(data['parsed_description'])
    sequences = tokenizer_obj.texts_to_sequences(data['parsed_description'])
    word_index = tokenizer_obj.word_index
    review_pad = pad_sequences(sequences, maxlen=max_length)
    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words,EMBEDDING_DIM))

    embeddings_index = {}
    f = open(os.path.join('','word_embedding_wordvec.txt'), encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close()   

    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    #Create the model
    model = Sequential()
    embedding_layer = Embedding(num_words,
                    output_dim=EMBEDDING_DIM,
                    embeddings_initializer=Constant(embedding_matrix),
                    input_length=max_length,
                    #ask_zero=True,
                    trainable=False)
    model.add(embedding_layer)
    model.add(GRU(units=32, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(19, activation='sigmoid'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    indices = np.arange(review_pad.shape[0])
    np.random.shuffle(indices)
    review_pad = review_pad[indices]
    categories =  data["category"][indices]
    num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

    X_train_pad = review_pad[:-num_validation_samples]
    y_train = categories[:-num_validation_samples]

    X_test_pad = review_pad[-num_validation_samples:]
    y_test = categories[-num_validation_samples:]
    
    history = model.fit(X_train_pad, y_train, epochs=25, batch_size=128,validation_data=(X_test_pad, y_test) ,verbose=2)

if __name__ == '__main__':
    print("Iam here")
    data = read_data()
    data_preproc = word2VecModel(data)
    data_preproc['category'] = data['category']
    data_preproc =  encodeLabel(data_preproc)
    data_preproc['parsed_description'] = data['parsed_description']
    makeTfModel(data_preproc) 


