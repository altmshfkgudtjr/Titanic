import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


path = "./model/"
graph_path = "./graph/"
trainSet = pd.read_csv("./data_set/ch_train.csv")
testSet = pd.read_csv("./data_set/ch_test.csv")


def modeling():
	train_label = trainSet['Survived']
	train_data = trainSet.drop(['Survived', 'PassengerId'], axis=1)

	model = keras.Sequential()
	model.add(keras.layers.Dense(32, activation='relu', input_shape=(4,)))
	model.add(keras.layers.Dense(16, activation='relu'))
	model.add(keras.layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

	cb_checkpoint = ModelCheckpoint(filepath=path+'titanic_model_3_{epoch:02d}.hdf5', monitor='accuracy' 
									, verbose=0, save_best_only=True, mode="auto", period=1)
	hist = model.fit(train_data, train_label, epochs = 200, batch_size = 1, shuffle=True
					, callbacks=[cb_checkpoint])
	graph(hist)


def best_model():
	model = load_model(path+'titanic_model_120.hdf5')

	test_data = testSet.drop(['PassengerId'], axis=1)
	print(model.predict_classes(test_data))
	submission = pd.DataFrame({
			"PassengerId": testSet["PassengerId"],
			"Survived": model.predict_classes(test_data).flatten()
		})

	print(submission)

	submission.to_csv(path+'submission.csv', index=False)


def graph(hist):
	train_label = trainSet['Survived']

	plt.figure(figsize=(12,8))
	plt.plot(hist.history['accuracy'])
	# plt.plot(hist.history['val_acc'])
	plt.plot(hist.history['loss'])
	# plt.plot(hist.history['val_loss'])
	plt.legend(['accuracy', 'loss'])
	plt.savefig(graph_path+"test_2.png")
	plt.show()

	plt.scatter(trainSet['PassengerId'], train_label, label='y_true')
	plt.scatter(trainSet['PassengerId'], model.predict(train_data), label='y_pred')
	plt.legend()
	plt.savefig(graph_path+"test_1.png")