from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

token_index = {}

tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples) # 단어 인덱스 구축

sequences = tokenizer.texts_to_sequences(samples)  # 무자열을 정수 인덱스의 리스트로 반환
print("sequence", sequences)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
print("onehot shape",one_hot_results.shape)
print("onehot result :",one_hot_results[:1,:10])


word_index = tokenizer.word_index

print(word_index)

print('%s개의 고유한 토큰을 찾았습니다.' % len(word_index))
