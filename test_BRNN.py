from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from pylab import mpl  
mpl.rcParams['font.sans-serif'] = ['SimHei']

max_features = 10000 # 特征单词的个数
maxlen = 500  # 在这么多单词之后截断文本

# 加载数据
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# 填充序列
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32, dropout=0.2, recurrent_dropout=0.2)))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train,
	                epochs=10, 
	                batch_size=128, 
	                validation_split=0.2)

# 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='训练精度')
plt.plot(epochs, val_acc, 'b', label='验证精度')
plt.title('训练和验证精度')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='训练损失')
plt.plot(epochs, val_loss, 'b', label='验证损失')
plt.title('训练和验证损失')
plt.legend()

plt.show()