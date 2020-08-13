'''
This is a controller that exposes endpoints for classification.
/: default endpoint to make sure the service is running
/classify: the service that takes key value pair as input and classify the data

'''
import flask
from flask import Flask, request, jsonify
import pandas as pd
from MultinomialNBClassifier import TransactionClassification
app = Flask(__name__)

@app.route('/')
def index():
    return "This is the classifier service. Please POST your request to /classify endpoint."

@app.route('/classify')
def classify():
    try:
        json_ = request.json
        print(json_)
        obj = TransactionClassification()
        classified_output = obj.classify(json_.get("key"))
        return jsonify({'Prediction': str(classified_output)})
    except:
        return jsonify({'Error': "Something wrong has happened."})
    else:
        print('Model not good.')
    return ('Model is not good.')


if __name__=='__main__':
    port = 12345
    app.run(port=port, debug=True)