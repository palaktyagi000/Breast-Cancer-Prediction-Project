# -*- coding: utf-8 -*-
"""
Created on Sun May 30 21:47:46 2021

@author: Palak Tyagi
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle 

app = Flask(__name__)
model = pickle.load(open('random_forest_classification_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI

    
    -------
    None.

    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if output == 1:
        return render_template('index.html', prediction_text='Breast Cancer is Diagnosed')
    else:
        return render_template('index.html', prediction_text="Don't Worry. You don't have Cancer")
        

    #return render_template('index.html', prediction_text='The output is {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
    
    


