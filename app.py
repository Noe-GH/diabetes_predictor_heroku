import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# Web app endpoint
@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    features_array = [np.array(features)]

    prediction = model.predict(features_array)[0]

    if prediction == 0:
        outcome = 'No diabetes'
    else:
        outcome = 'Diabetes'

    return render_template('index.html', prediction_text='The predicted diagnosis is: {}'.format(outcome))

# API endpoint
@app.route('/predict_api/')
def predict_api():
    '''
    For returning results in json format
    '''
    model = pickle.load(open('model.pkl', 'rb'))
    pregnancies = request.args.get('pregnancies')
    glucose = request.args.get('glucose')
    blood_press = request.args.get('blood_press')
    skin_thick = request.args.get('skin_thick')
    insulin = request.args.get('insulin')
    bmi = request.args.get('bmi')
    d_pedigree_f = request.args.get('d_pedigree_f')
    age = request.args.get('age')

    df = pd.DataFrame({'pregnancies':[pregnancies], 'glucose':[glucose],
                       'blood_press':[blood_press], 'skin_thick':[skin_thick],
                       'insulin':[insulin], 'bmi':[bmi],
                       'd_pedigree_f':[d_pedigree_f], 'age':[age]})

    prediction = model.predict(df)[0]

    if prediction == 0:
        outcome = 'No diabetes'
    else:
        outcome = 'Diabetes'

    return jsonify({'diabetes':outcome})


if __name__ == "__main__":
    app.run(debug=True)