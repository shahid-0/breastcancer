import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle

app = Flask(__name__)
model = pickle.load(open('D:/ML project/env/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(float(x)) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    if (output==0):
        return "The patient has a Cancer. Please go and meet Doctor and take treatment."
    else:
        return "The patient has No Cancer."
    return render_template('index.html', prediction_text='Cancer if yes=0, no=1 {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)