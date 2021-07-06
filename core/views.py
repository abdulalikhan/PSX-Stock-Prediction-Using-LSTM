from django.shortcuts import render
from .models import Prediction

import numpy as np
import pandas as pd
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
# Create your views here.


def index(request):
    return render(request, 'index.html')


def disclaimer(request):
    return render(request, 'disclaimer.html')


def predict(request):
    if request.method == 'POST':
        company_symbol = request.POST['company_title']
        if (company_symbol == 'ACPL'):
            company_name = "Attock Cement [ACPL]"
        elif (company_symbol == 'BWCL'):
            company_name = "Bestway Cement [BWCL]"
        elif (company_symbol == 'CHCC'):
            company_name = "Cherat Cement [CHCC]"
        elif (company_symbol == 'DCL'):
            company_name = "Deewan Cement [DCL]"
        elif (company_symbol == 'DGKC'):
            company_name = "D.G. Khan Cement [DGKC]"
        elif (company_symbol == 'DNCC'):
            company_name = "Dandot Cement [DNCC]"
        elif (company_symbol == 'FCCL'):
            company_name = "Fauji Cement [FCCL]"
        elif (company_symbol == 'FECTC'):
            company_name = "Fecto Cement [FECTC]"
        elif (company_symbol == 'FLYNG'):
            company_name = "Flying Cement [FLYNG]"
        elif (company_symbol == 'GWLC'):
            company_name = "Gharibwal Cement [GWLC]"
        elif (company_symbol == 'JVDC'):
            company_name = "Javedan Corporation [JVDC]"
        elif (company_symbol == 'KOHC'):
            company_name = "Kohat Cement [KOHC]"
        elif (company_symbol == 'LUCK'):
            company_name = "Lucky Cement [LUCK]"
        elif (company_symbol == 'MLCF'):
            company_name = "Maple Leaf Cement [MLCF]"
        elif (company_symbol == "PIOC"):
            company_name = "Pioneer Cement [PIOC]"
        elif (company_symbol == "POWER"):
            company_name = "Power Cement [POWER]"
        elif (company_symbol == "SMCPL"):
            company_name = "Safe Mix Concrete [SMCPL]"
        elif (company_symbol == "THCCL"):
            company_name = "Thatta Cement"
        company = company_symbol

        data = pd.read_csv("data/train/"+company.lower()+".csv")
        data['TIME'] = pd.to_datetime(data['TIME'])
        start = data.TIME.min()
        end = data.TIME.max()
        mask = (data['TIME'] > start) & (data['TIME'] <= end)
        data = data.loc[mask]

        # Prepare Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['CLOSE'].values.reshape(-1, 1))

        # How many days to look into the past
        # How many days back should the prediction be based on
        prediction_days = 60

        # Load the model
        model_fname = 'models/model_' + company.lower() + '.h5'
        model = load_model(model_fname)

        # Load test data
        test_data = pd.read_csv("data/test/"+company.lower()+".csv")
        test_data['TIME'] = pd.to_datetime(test_data['TIME'])
        test_start = test_data.TIME.min()
        test_end = test_data.TIME.max()

        mask = (test_data['TIME'] > test_start) & (
            test_data['TIME'] <= test_end)
        test_data = test_data.loc[mask]

        total_dataset = pd.concat((data['CLOSE'], test_data['CLOSE']), axis=0)

        model_inputs = total_dataset[len(
            total_dataset)-len(test_data)-prediction_days:].values
        model_inputs = model_inputs.reshape(-1, 1)
        model_inputs = scaler.transform(model_inputs)

        # Calculate number of days between the last row and today
        tomorrow = dt.datetime.today()
        today = dt.datetime.now()
        predicted_prices = []
        days = []
        for i in range(1, 8):
            day_gap = (test_end - (today+dt.timedelta(days=i))).days

            # Predict next day
            real_data = [model_inputs[len(
                model_inputs) + day_gap - prediction_days:len(model_inputs)+day_gap, 0]]
            real_data = np.array(real_data)
            real_data = np.reshape(
                real_data, (real_data.shape[0], real_data.shape[1], 1))

            prediction = model.predict(real_data)
            prediction = scaler.inverse_transform(prediction)
            tomorrow = tomorrow+dt.timedelta(days=1)
            #print("Predicted Stock Price on {0}: RS {1}".format(tomorrow.strftime('%d-%m-%Y'), str(prediction[0, 0])))
            days.append(tomorrow.strftime('%d-%m-%Y'))
            predicted_prices.append(round(prediction[0, 0], 2))

        predictions = []
        for i in range(0, 7):
            prediction = Prediction(days[i], predicted_prices[i])
            predictions.append(prediction)
        return render(request, 'predict.html', {'company_title': company_name, 'predictions': predictions, 'days': days, 'predicted_prices': predicted_prices})
    else:
        return render(request, 'index.html')
