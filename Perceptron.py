# Build a perceptron classifier in Python

# Use keras package
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

# Set batch size for training, epochs and number of classes
batch_size = 128
num_classes = 10
epochs = 20 

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape data for training
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Divide x by 255 so that it's in the range of (0, 1) for higher training speed
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Categorize y
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build a perceptron model
model_p = Sequential()
model_p.add(Dense(num_classes, activation='softmax', input_shape = (784,)))
model_p.summary()

# Compile model 
model_p.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Train model
history_p = model_p.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
                    
# Evaluate model
score_p = model_p.evaluate(x_test, y_test, verbose=0)
