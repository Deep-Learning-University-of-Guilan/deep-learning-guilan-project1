import numpy as np
import pandas as pd
from neuralnet.layers import FCLayer, SigmoidLayer
from neuralnet.losses import BinaryCrossEntropy
from neuralnet.network import NeuralNetwork
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('tests/data/Dataset.csv')
df.loc[(df["Class"] == "M") | (df["Class"] == "H"), "Class"] = 1
df.loc[df["Class"] == "L", "Class"] = 0
# X: Features, y: Classes
y = np.array(df['Class'])
X = np.array(pd.get_dummies(df.iloc[:, :-1])).astype("float")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=12)
normalizer = StandardScaler()
X_train = normalizer.fit_transform(X_train)
X_val = normalizer.transform(X_val)

print('Number of dataset: ', len(X))
print('Number of train set: ', len(X_train))
print('Number of validation set: ', len(X_val))

model = NeuralNetwork([
    FCLayer(X_train.shape[1], 32, random=True),
    SigmoidLayer(),
    FCLayer(32, 32, random=True),
    SigmoidLayer(),
    FCLayer(32, 32, random=True),
    SigmoidLayer(),
    FCLayer(32, 1, random=True),
    SigmoidLayer(),
])

model.compile(BinaryCrossEntropy(), ["accuracy"])

losses, accs, losses_val, accs_val = model.fit(X_train,
                                               y_train,
                                               EPOCHS=400,
                                               learning_rate=0.01,
                                               validation_data=(X_val, y_val)
                                               )
