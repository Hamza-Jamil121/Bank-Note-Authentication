# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np


filename = 'classifier.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('Responsive.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        var = int(request.form['variance'])
        ske = int(request.form['skewness'])
        curt= int(request.form['curtosis'])
        entro = int(request.form['entropy'])
      
        data = np.array([[var,ske,curt,entro]])
        my_prediction = classifier.predict(data)
        
        if my_prediction == 1:
            
            res_val = "is not Authentic"
        else:
            
            res_val = "is Authentic"
        
        return render_template('Responsive.html',prediction_text='Bank Note {}'.format(res_val))

if __name__ == '__main__':
	app.run(debug=True)