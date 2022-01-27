import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
import bz2
import pickle
import _pickle as cPickle

def decompress_pickle(CarSelling):
    random_model=bz2.BZ2File(CarSelling,'rb')
    random_model=cPickle.load(random_model)
    return random_model

app = Flask(__name__)
model1=decompress_pickle('CarSelling.pbz2')
# model = pickle.load(open('CarSelling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model1.predict(final_features)
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Price Predicition is $ {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)