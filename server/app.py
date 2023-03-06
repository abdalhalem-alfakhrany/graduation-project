from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')

classifier = pickle.load(
    open('..//Ai Model//logisticRegressionClassifier.pkl', 'rb'))
scaler = pickle.load(open('..//Ai Model//framinghamDataSetScaler.pkl', 'rb'))


@app.route('/')
def getSlash():
    return render_template('index.html')


@app.route('/get_prediction', methods=['POST'])
def getPrediction():
    json = request.json
    transformedData = scaler.transform([[
        json['gender'], json['age'],
        json['education'], json['currentSmoker'],
        json['cigs_pre_day'], json['BPMeds'],
        json['prevalentStroke'], json['prevalentHypertension'],
        json['totalCholesterol'], json['systolicBP'],
        json['diabetes'], json['diastolicBP'],
        json['BMI'], json['hartRate'], json['glucose'],
    ]])

    return str(classifier.predict(transformedData)[0])
