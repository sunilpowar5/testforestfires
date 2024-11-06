
import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge regressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
stadard_scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('WS'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DDC = float(request.form.get('DDC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = stadard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DDC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home1.html',results=result[0])

    
    else:
        return render_template('home1.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')