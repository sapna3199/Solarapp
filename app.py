from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from sklearn.tree import DecisionTreeClassifier
import joblib

app = Flask(__name__)

file = open('G:/project2/finalized_model.pkl', 'rb')
model = pickle.load(file)
file.close()

@app.route('/')
def home():
    return(render_template('index.html'))

@app.route('/predict',methods=['POST'])
def predict():
    
    # input
    input_features = [float(x) for x in request.form.values()]
    feature_value = np.array(input_features)
    
    # output
    output = model.predict([feature_value])[0].round(0)
    
    if output == 0:
        return render_template('index.html', prediction_text='Solar panel is faulty.')
    else:
        return render_template('index.html', prediction_text='Solar panel is nort faulty.')
    
    return render_template('index.html')



if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)
