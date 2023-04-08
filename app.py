import numpy as np 
import pandas as pd 
from flask import Flask,request,render_template
import pickle
#from joblib import load 
import os

app = Flask(__name__)
model = pickle.load(open('Visalr.pkl','rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction')
def prediction():
    return render_template('index.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/prediction/y_predict',methods=['POST'])
def y_predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['FULL_TIME_POSITION','PREVAILING_WAGE','YEAR','SOC_N']
    df = pd.DataFrame(features_value, columns = features_name)
    output = model.predict(df)
    output = np.argmax(output)

    if prediction==0:
        output="Certified"
    else:
        output="denied"
    print(output)
    return render_template('index.html', prediction_text='   {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
    