import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, LSTM, Dense, Dropout, GRU
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
import tensorflow as tf

df = pd.read_csv("datset_depat2.csv", sep=";")

df["semaine"] = pd.to_datetime(df["semaine"])
df.set_index("semaine", inplace=True)

df.drop(columns=["T", 'Tp','Td'], inplace=True)

df['Ti'] = pd.to_numeric(df['Ti'])

max = df.values.max()
split = int(len(df)*0.8)
print(max, split)

train = df[:split].values / max
test = df[split:].values / max
input_length = 4
features_length = 1
train_generator = TimeseriesGenerator(train, train, length=input_length, batch_size=1)
test_generator = TimeseriesGenerator(test, test, length=input_length, batch_size=1)

x, y = train_generator[0]
print(x.shape)
print(y.shape)
input_length = x.shape[1]
features_length = x.shape[2]

lstm_model = Sequential()

lstm_model.add(InputLayer(input_shape=(input_length, features_length)))
lstm_model.add(LSTM(50, activation="tanh", recurrent_activation="sigmoid", return_sequences=False))
lstm_model.add(Dropout(0.3))

lstm_model.add(Dense(features_length, activation="sigmoid"))

optmizer = tf.keras.optimizers.Adam(learning_rate=0.005)

lstm_model.compile(optimizer=optmizer, loss="mse", metrics=["mae"])
lstm_model.summary()
def scheduler(epoch, lr):
  return lr*tf.math.exp(-0.2)
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = lstm_model.fit(train_generator, epochs=19, validation_data=test_generator,callbacks=[learning_rate_scheduler, early_stopping], verbose=1)
print(test_generator[0])
y_true = history.history['val_loss'] 

time_steps = range(0, len(y_true))
# Tracé des données réelles et prédites

plt.plot(time_steps,y_true, label='Données réelles')

# Personnalisation du graphique
plt.xlabel('Temps')
plt.ylabel('Valeur')
plt.title('Prédiction temporelle')
plt.legend()

# # Affichage du graphique
plt.show()

