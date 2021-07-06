# PSX Stock Price Prediction Using Long Short-Term Memory (LSTM) Neural Networks

This web application forecasts stock prices for cement sector companies listed on the Pakistan Stock Exchange (PSX).

## Live Application
The web application is hosted at [https://psx-forecast.herokuapp.com/](https://psx-forecast.herokuapp.com/)

## Screenshots
![Home Page](https://github.com/abdulalikhan/PSX-Stock-Prediction-Using-LSTM/blob/main/1.png?raw=true)
![Selected a Company](https://github.com/abdulalikhan/PSX-Stock-Prediction-Using-LSTM/blob/main/2.png?raw=true)
![7-Day Forecast](https://github.com/abdulalikhan/PSX-Stock-Prediction-Using-LSTM/blob/main/3.png?raw=true)
![Predicted Stock Prices](https://github.com/abdulalikhan/PSX-Stock-Prediction-Using-LSTM/blob/main/4.png?raw=true)

## Built Using
- :gear: Django
- :snake: Python
- :brain: Tensorflow
- :panda_face: Pandas
- :100: Numpy
- :zap: Scikit-Learn
- :art: Bootstrap

## Methodology

- All LSTM models were trained on closing price data obtained from the Pakistan Stock Exchange's official website. 
- The data set was divided into two parts. 
  - The part uptil 8th October 2020 was used to train the models, and the part from 9th October 2020 to 2nd July 2021 was used for testing purposes.
  - After testing all the models, the testing data was combined with the training data to train all the models with a larger data set.

### Mean Absolute Percentage Errors

To determine the accuracy of the models, I calculated the Mean Absolute Percentage Error using the predicted and actual closing price values for the testing data.

| Company                   | Mean Absolute Percentage Error  |
| ------------------------- | -------------------------------:|
| Attock Cement (ACPL)      | 8.71%                           |
| Bestway Cement (BWCL)     | 4.94%                           |
| Cherat Cement (CHCC)      | 14.99%                          |
| Dewan Cement (DCL)        | 10.70%                          |
| D.G. Khan Cement (DGKC)   | 8.69%                           |
| Dandot Cement (DNCC)      | 28.00%                          |
| Fauji Cement (FCCL)       | 7.95%                           |
| Fecto Cement (FECTC)      | 10.44%                          |
| Flying Cement (FLYNG)     | 27.82%                          |
| Gharibwal Cement (GWLC)   | 18.26%                          |
| Javedan Cement (JVDC)     | 29.50%                          |
| Kohat Cement (KOHC)       | 5.98%                           |
| Lucky Cement (LUCK)       | 14.25%                          |
| Maple Leaf Cement (MLCF)  | 7.48%                           |
| Pioneer Cement (PIOC)     | 15.70%                          |
| Power Cement (POWER)      | 9.23%                           |
| Safemix Concrete (SMCPL)  | 15.39%                          |
| Thatta Cement (THCCL)     | 9.86%                           |
