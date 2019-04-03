import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
import numpy as np


# 데이터 준비
datasetDir = '/home/devel/tensorflow-tutorials/playground/datasets'
imdbDir = os.path.join(datasetDir, 'aclImdb')
trainDir = os.path.join(imdbDir, 'train')

labels = []
texts = []

for labelType in ['neg', 'pos']:
    dirName = os.path.join(trainDir, labelType)
    for fname in os.listdir(dirName):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dirName, fname), encoding="utf8")
            texts.append(f.read())
            f.close()
            if labelType == 'neg':
                labels.append(0)
            else :
                labels.append(1)

# 데이터 토큰화
maxWords = 10000 # 1만개 단어
tokenizer = Tokenizer(num_words=maxWords) # num_words 세팅이 동작함??
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

wordIndex = tokenizer.word_index
print('토큰 %s개' % len(wordIndex))

maxLen = 100 # 문장당 단어 100개
data = pad_sequences(sequences, maxlen=maxLen)
labels = np.asanyarray(labels)
print('data tensor shape : ', data.shape)
print('labels tensor shape : ', labels.shape)

# imdb가 긍정 샘플 나온후 부정 샘플로 나오도록 구성되어 있으므로 shuffle 필요
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

trainingSamples = 200 # 훈련 샘플 200개
validationSamples = 10000
xTrain = data[:trainingSamples]
yTrain = labels[:trainingSamples]
xVal = data[trainingSamples : trainingSamples+validationSamples]
yVal = labels[trainingSamples : trainingSamples+validationSamples]


embeddingsIndex = {}
f = open(os.path.join(datasetDir, 'glove.6B.50d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asanyarray(values[1:], dtype='float32')
    embeddingsIndex[word] = coefs
f.close()

print("%s개의 단어 벡터를 찾았습니다." % len(embeddingsIndex))

# Glove 단어 임베딩 행렬 준비
embeddingsDim = 50
embeddingsMatrix = np.zeros((maxWords, embeddingsDim))
for word, i in wordIndex.items():
    if i<maxWords:
        embeddingsVector = embeddingsIndex.get(word)
        if embeddingsVector is not None:
            embeddingsMatrix[i] = embeddingsVector

# 모델 정의하기
model = Sequential()
model.add(Embedding(maxWords, embeddingsDim, input_length=maxLen, name='layerEmbedding'))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()


model.layers[0].set_weights([embeddingsMatrix])
model.layers[0].trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(xTrain, yTrain, epochs=10, batch_size=32, validation_data=(xVal,yVal))
model.save_weights(os.path.join(datasetDir,'pre_trained_glove_model.h5'))
