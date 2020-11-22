import requests
import pandas as pd
import scipy
import numpy as np
import sys
from scipy import stats


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    url_content = response.content
    file = open('train.csv', 'wb')
    file.write(url_content)
    file.close()
    df = pd.read_csv('train.csv', header = None).T
    h = df.iloc[0]
    df.drop([0],axis = 0, inplace = True)
    df.columns = h
    x_train = np.asarray(df.area.astype('float'))
    y_train = np.asarray(df.price.astype('float'))
    
    b, a, r_val, p_val, std_error = stats.linregress(x_train,y_train)

    #predictions
    y_pred = a + b*area
    
    return y_pred
       
if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = np.array(list(validation_data.keys()))
    prices = np.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = np.sqrt(np.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
