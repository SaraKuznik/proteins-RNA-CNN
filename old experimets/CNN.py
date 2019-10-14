import keras
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.models import Model

import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

import random
# random.seed(42)

X = []
Y = []
structure_ids = []
for line in open('./structures lists/structures human.txt', 'r'):
    line = line.strip('\n')
    structure_ids.append(line)
for line in open('./structures lists/structures ecoli.txt', 'r'):
    line = line.strip('\n')
    structure_ids.append(line)
random.shuffle(structure_ids)
print(len(structure_ids))

for structure_id in structure_ids:
    protein = np.load('./voxelized data/' + structure_id + '_protein.npy', mmap_mode='r')
    rna = np.load('./voxelized data/' + structure_id + '_rna.npy', mmap_mode='r')
    X.append(protein[:10])
    X.append(protein[-10:])
    # rna = list(map(sum, rna))
    Y.append(rna[:10])
    Y.append(rna[-10:])

X = np.concatenate(X)
Y = np.concatenate(Y)
Y[Y > 0] = 1

num_train = 4840
X_train = X[:num_train]
Y_train = Y[:num_train]
X_test = X[num_train:]
Y_test = Y[num_train:]
print(X_train.shape, X_test.shape)


ins = Input((5, 5, 5, 3))
con1 = Conv3D(filters=8, kernel_size=(3, 3, 3), padding='same', activation='relu')(ins)
con2 = Conv3D(filters=16, kernel_size=(3, 3, 3), padding='same', activation='relu')(con1)
maxp3 = MaxPool3D(pool_size=(2, 2, 2))(con2)
con4 = Conv3D(filters=32, kernel_size=(3, 3, 3), padding='same', activation='relu')(maxp3)
con5 = Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same', activation='relu')(con4)
maxp6 = MaxPool3D(pool_size=(2, 2, 2))(con5)
batch = BatchNormalization()(maxp6)
flat = Flatten()(batch)
# dens1 = Dense(units=2048, activation='relu')(flat)
# drop1 = Dropout(0.6)(dens1)
dens2 = Dense(units=1024, activation='relu')(flat)
drop2 = Dropout(0.4)(dens2)
outs = Dense(units=5, activation='sigmoid')(drop2)
model = Model(inputs=ins, outputs=outs)
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adadelta(lr=1))

model.summary()

model.fit(x=X_train, y=Y_train, batch_size=200, epochs=50)
# print(model.evaluate(X_test, Y_test, verbose=0, batch_size=100))
Y_pred = model.predict(X_test, batch_size=100)
print(Y_pred[:10])
# Y_pred = np.zeros(Y_test.shape)
mse = sklearn.metrics.mean_squared_error(Y_test, Y_pred)
r2 = sklearn.metrics.r2_score(Y_test, Y_pred)
print(f'MSE: {mse}, R2: {r2}')

fig = plt.figure(figsize=plt.figaspect(0.5))

for i in range(5):
    ax = fig.add_subplot(2, 3, i+1)
    ax.plot(Y_pred[:,i], Y_test[:,i], 'o')
    ax.set_xlabel('Y predicted')
    ax.set_ylabel('Y true')

ax = fig.add_subplot(2, 3, 6)
ax.plot(list(map(sum, Y_pred)), list(map(sum, Y_test)), 'ro')
fig.suptitle(f'MSE: {np.round(mse, 2)}, R2: {np.round(r2, 2)}')
plt.show()
