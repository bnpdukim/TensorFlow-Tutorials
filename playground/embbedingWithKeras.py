from keras.datasets import imdb
from keras import preprocessing
from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding


maxFeatures = 10000
maxLen = 20

(xTrain, yTrain), (xTest, yTest) = imdb.load_data(num_words=maxFeatures)
xTrain = preprocessing.sequence.pad_sequences(xTrain, maxlen=maxLen)
xTest = preprocessing.sequence.pad_sequences(xTest, maxlen=maxLen)

print('xTrain shape : ', xTrain.shape) # 25000,20
print('xTest shape : ', xTest.shape) # 25000,20

model = Sequential()
model.add(Embedding(maxFeatures, 8, input_length=maxLen, name='layerEmbedding'))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()

history = model.fit(xTrain, yTrain, epochs=10, batch_size=32, validation_split=0.2)
embeddingWeigts = model.get_layer('layerEmbedding').get_weights()
print(embeddingWeigts.shape)
