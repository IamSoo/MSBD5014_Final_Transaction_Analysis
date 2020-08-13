import pandas
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

class  TransactionClassification:
    def __init__(self):
        self.pickle = pickle

    def load_model(self, model_file_name):
        self.model, self.vectorizer, self.encoder = self.pickle.load(open(model_file_name,'rb'))

    def count_vectorizer(self, input):
        query_data = self.vectorizer.transform(input)
        return query_data

    def predict_output(self, query_data):
        return self.model.predict(query_data)

    def encode_data(self, input):
        encoded_data = self.encoder.fit_transform(input)
        return encoded_data

    def decode_predicted_data(self, predicted_value):
        predicted_label = self.encoder.inverse_transform(predicted_value)
        return predicted_label

    def classify(self, input):
        self.load_model('MultinomialNBClassifier.sav')
        data = [input]
        query_data = self.count_vectorizer(data)
        encoded_output = self.predict_output(query_data)
        output = self.decode_predicted_data(encoded_output)
        return output

   



