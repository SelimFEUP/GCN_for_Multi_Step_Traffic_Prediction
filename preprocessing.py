import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(path):
    data = pd.read_csv(path, index_col=0, parse_dates=True)
    data.fillna(method='ffill', inplace=True)
    data.interpolate(method='linear', inplace=True)
    return data.values

path = 'transformed_data.csv'
data = load_data(path)
print(data.shape)

def create_seq(data, input_steps, output_steps):
    x, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        x.append(data[i:(i+input_steps), :])
        y.append(data[(i+input_steps):(i+input_steps+output_steps), :])
    return np.array(x), np.array(y)
x, y = create_seq(data, 24, 24)
print(x.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape)


scaler = MinMaxScaler()
num_feature = x_train.shape[2]
x_train_reshaped = x_train.reshape(-1, num_feature)
x_test_reshaped = x_test.reshape(-1, num_feature)
x_train_scaled = scaler.fit_transform(x_train_reshaped).reshape(x_train.shape)
x_test_scaled = scaler.transform(x_test_reshaped).reshape(x_test.shape)
y_train_reshaped = y_train.reshape(-1, num_feature)
y_test_reshaped = y_test.reshape(-1, num_feature)
y_train_scaled = scaler.fit_transform(y_train_reshaped).reshape(y_train.shape)
y_test_scaled = scaler.transform(y_test_reshaped).reshape(y_test.shape) 

print(x_train_scaled.shape, y_train_scaled.shape)
print(x_test_scaled.shape, y_test_scaled.shape)

