import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import accuracy_score

import numpy as np

npz_kw = np.load('./v-melsp-test.npz')

x_test = npz_kw['x']
y_test = npz_kw['y']

testdata = x_test[..., None]

model = load_model("model_second_cnn.h5")
ret = model.predict(testdata, batch_size=1)
for i in range(len(ret)):
    print("{:.2%}, {:.2%}".format(ret[i][0], ret[i][1]))
