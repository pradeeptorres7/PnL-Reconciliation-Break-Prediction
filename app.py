from flask import Flask
from flask import request
from flask import render_template
from flask import send_file

import pandas as pd
import os
import joblib

app = Flask(__name__)

model = joblib.load(
"models/break_model.pkl"
)

@app.route('/')
def home():

    return render_template(
        'index.html'
    )

@app.route('/predict',methods=['POST'])
def predict():

    values=[

        float(request.form['DV01']),
        float(request.form['Vega']),
        float(request.form['Gamma']),
        float(request.form['MarketMove']),
        float(request.form['VaR']),
        float(request.form['ResidualPnL']),
        float(request.form['TradeCount'])
    ]

    """
    result=model.predict([values])

    if result[0]==1:
        prediction="Reconciliation Break Likely"
    else:
        prediction="No Reconciliation Break"

    return render_template(
        'index.html',
        prediction_text=prediction 
        """
    
    result = model.predict([values])

    if result[0]==1:
        prediction="Reconciliation Break Likely"
    else:
        prediction="No Reconciliation Break"

    prob = model.predict_proba([values])

    prediction_prob = round(prob[0][1] * 100,2)

    return render_template(

    'index.html',

    prediction_text=prediction,

    probability_text=f'Prediction Confidence : {prediction_prob}%'

)

@app.route('/Bulk_upload',
           methods=['POST'])
def Bulk_upload():

    file = request.files['file']

    filepath = os.path.join(
        'uploads',
        file.filename
    )

    file.save(filepath)

    df = pd.read_csv(filepath)

    predictions = model.predict(df)

    probabilities = model.predict_proba(df)

    #df['Prediction'] = predictions

    #df['Probability'] = probabilities[:,1]

    df['Prediction'] = [
        'Reconciliation Break' if p == 1
        else 'No Reconciliation Break'
        for p in predictions
                    ]

    df['Probability'] = (probabilities[:, 1] * 100).round(1).astype(str) + '%'

    output_file = 'uploads/output.csv'

    df.to_csv(
        output_file,
        index=False
    )

    return send_file(
        output_file,
        as_attachment=True
    )

if __name__=='__main__':
    app.run(debug=True)