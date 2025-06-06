import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Add, Dense, Input
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

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


class GraphConvolution(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim

    def build(self, input_shape):
        num_nodes = input_shape[-1]
        self.weight = self.add_weight(shape=(num_nodes, self.output_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        self.adj = self.add_weight(shape=(num_nodes, num_nodes),
                                   initializer='glorot_uniform',
                                   trainable=True,
                                   name='adjacency_matrix')

    def call(self, inputs):
        # inputs shape: (batch, time, nodes)
        support = tf.einsum('bti,io->bto', inputs, self.weight)
        output = tf.einsum('ij,bto->bti', self.adj, support)
        return output
        

def build_gcn_model(input_steps, output_steps, num_nodes, num_heads=4):
    inputs = Input(shape=(input_steps, num_nodes))  # (batch, time, nodes)

    # Graph Convolution Layer
    gc1 = GraphConvolution(64)(inputs)              # (batch, time, 64)
    gc1 = tf.keras.layers.ReLU()(gc1)

    # LSTM Layers
    lstm_out = tf.keras.layers.LSTM(312, return_sequences=True)(gc1)  # (batch, time, 312)
    lstm_out = tf.keras.layers.LSTM(128, return_sequences=True)(lstm_out)  # (batch, time, 128)

    # Multi-Head Attention (Self-attention over time)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=32)(lstm_out, lstm_out)
    attn_output = Add()([lstm_out, attn_output])  # Residual connection
    attn_output = LayerNormalization()(attn_output)

    # Optional: Feed-forward transformation after attention
    ff = Dense(128, activation='relu')(attn_output)
    ff = LayerNormalization()(ff)

    # Slice the last output_steps from the sequence
    sliced = ff[:, -output_steps:, :]  # (batch, output_steps, features)

    # Final dense projection to predict traffic state per node
    outputs = tf.keras.layers.TimeDistributed(Dense(num_nodes))(sliced)  # (batch, output_steps, num_nodes)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

    
input_steps = 24
output_steps = 24
num_nodes = x_train.shape[2]

model = build_gcn_model(input_steps, output_steps, num_nodes)
model.summary()

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=7, mode='min')
mc = tf.keras.callbacks.ModelCheckpoint('fast_api/gcn.weights.h5', monitor='val_mae', verbose=5, save_best_only=True, 
          mode='min', save_weights_only=True)

history = model.fit(x_train_scaled, y_train_scaled, validation_split=0.1, epochs=100, batch_size=32, callbacks=[mc, early_stopping])

# Predict
y_pred_scaled = model.predict(x_test_scaled)

# Inverse transform
#y_pred_reshaped = y_pred_scaled.reshape(-1, num_nodes)
#y_test_reshaped = y_test_scaled.reshape(-1, num_nodes)
#y_pred = scaler.inverse_transform(y_pred_reshaped).reshape(y_pred_scaled.shape)
#y_test = scaler.inverse_transform(y_test_reshaped).reshape(y_test_scaled.shape)

# Evaluation metrics
mae = mean_absolute_error(y_test_scaled.flatten(), y_pred_scaled.flatten())
rmse = root_mean_squared_error(y_test_scaled.flatten(), y_pred_scaled.flatten())

print(f"Test MAE: {mae:.4f}, RMSE: {rmse:.4f}")

