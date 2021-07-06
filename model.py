import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
companies = ['ACPL','BWCL','CHCC','DCL','DGKC','DNCC','FCCL','FECTC','FLYNG',
             'GWLC','JVDC','KOHC','LUCK','MLCF','PIOC','POWER','SMCPL','THCCL']
#company_index = 0
for company_index in range(0, len(companies)):
    company = companies[company_index]

    data = pd.read_csv('data/train/'+company.lower()+".csv")
    data['TIME'] = pd.to_datetime(data['TIME'])
    start = data.TIME.min()
    end = data.TIME.max()
    mask = (data['TIME'] > start) & (data['TIME'] <= end)
    data = data.loc[mask]

    # Prepare Data
    # LSTM are sensitive to the scale of the data, so we aply MinMax Scaler
    # Transform values into 0 to 1 (Scale down to 0-1)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data['CLOSE'].values.reshape(-1,1))

    # How many days to look into the past
    # How many days back should the prediction be based on
    prediction_days = 60

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build the model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1)) # Prediction of the next closing value

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    model_fname = 'model_' + company.lower() + '.h5'
    model.save(model_fname)
    '''
    # Test the model accuracy on existing data

    # Load test data
    test_data = pd.read_csv('data/test/'+company.lower()+".csv")
    test_data['TIME'] = pd.to_datetime(test_data['TIME'])
    test_start = test_data.TIME.min()
    test_end = test_data.TIME.max()

    # Calculate number of days between the last row and today
    day_gap = (test_end - (dt.datetime.now()+dt.timedelta(days=1))).days

    mask = (test_data['TIME'] > test_start) & (test_data['TIME'] <= test_end)
    test_data = test_data.loc[mask]
    actual_prices = test_data['CLOSE'].values

    total_dataset = pd.concat((data['CLOSE'], test_data['CLOSE']), axis=0)

    model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Made predictions on test data

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Measure absolute error
    abs_error = np.absolute((actual_prices - predicted_prices)/actual_prices)
    avg_error = np.mean(abs_error)
    avg_error = avg_error*100
    print("Absolute Percentage Relative Error for {0}: {1}%".format(company, avg_error))
    
    # Plot the test predictions
    plt.plot(actual_prices, color='black', label="Actual {0} Price".format(company))
    plt.plot(predicted_prices, color='green', label="Predicted {0} Price".format(company))
    plt.title("{0} Share Price".format(company))
    plt.xlabel('Days')
    plt.ylabel("{0} Share Price".format(company))
    plt.legend()
    #plt.show()
    plt.savefig(company)
    plt.close()
    # Predict next day
    real_data = [model_inputs[len(model_inputs) + day_gap - prediction_days:len(model_inputs)+day_gap, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    tomorrow = dt.datetime.today()+dt.timedelta(days=1)
    print("Predicted Stock Price on {0}: RS {1}".format(tomorrow.strftime('%d-%m-%Y'), str(prediction[0,0])))
    '''
