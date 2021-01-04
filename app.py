# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
cv1=pickle.load(open('tranform.pkl','rb'))
pkl_file = 'Pickle_RL_Model.pkl'
with open(pkl_file, 'rb') as file:
    pickle_model = pickle.load(file)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering the analysis result on html
    '''
    if request.method == 'POST':
	movie_review = request.form['text']
	vect = cv1.transform([movie_review])[0].toarray()
	#final_features = [np.array(int_features)]
	prediction = pickle_model.predict(vect)

#    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='The following comment is $ {}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)