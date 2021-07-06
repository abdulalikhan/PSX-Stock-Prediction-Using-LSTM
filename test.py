import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
companies = ['ACPL','BWCL','CHCC','DCL','DGKC','DNCC','FCCL','FECTC','FLYNG',
             'GWLC','JVDC','KOHC','LUCK','MLCF','PIOC','POWER','SMCPL','THCCL']
company_index = 17
company = companies[company_index]

data = pd.read_csv("data/train/"+company.lower()+".csv")
data['TIME'] = pd.to_datetime(data['TIME'])
start = data.TIME.min()
end = data.TIME.max()
mask = (data['TIME'] > start) & (data['TIME'] <= end)
data = data.loc[mask]

# Prepare Data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['CLOSE'].values.reshape(-1,1))

# How many days to look into the past
# How many days back should the prediction be based on
prediction_days = 60

# Load the model
model_fname = 'model_' + company.lower() + '.h5'
model = load_model(model_fname)

''' Test the model accuracy on existing data '''

# Load test data
test_data = pd.read_csv("data/test/"+company.lower()+".csv")
test_data['TIME'] = pd.to_datetime(test_data['TIME'])
test_start = test_data.TIME.min()
test_end = test_data.TIME.max()

mask = (test_data['TIME'] > test_start) & (test_data['TIME'] <= test_end)
test_data = test_data.loc[mask]

total_dataset = pd.concat((data['CLOSE'], test_data['CLOSE']), axis=0)

model_inputs = total_dataset[len(total_dataset)-len(test_data)-prediction_days:].values
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
    real_data = [model_inputs[len(model_inputs) + day_gap - prediction_days:len(model_inputs)+day_gap, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1],1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)
    tomorrow = tomorrow+dt.timedelta(days=1)
    print("Predicted Stock Price on {0}: RS {1}".format(tomorrow.strftime('%d-%m-%Y'), str(prediction[0,0])))
    days.append(tomorrow)
    predicted_prices.append(prediction[0,0])

# Plot the test predictions
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
plt.plot(days,predicted_prices, color='green', label="Predicted {0} Price".format(company))
plt.title("{0} Share Price".format(company))
plt.xlabel('Days')
plt.ylabel("{0} Share Price".format(company))
plt.legend()
plt.gcf().autofmt_xdate()
#plt.show()
plt.savefig(company)
plt.close()
