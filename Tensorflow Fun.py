#%%
from __future__ import print_function
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten,  Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

#%%
now = datetime.datetime.now
batch_size = 128
num_classes = 5
epochs = 6
img_rows, img_cols = 28, 28
filters = 64
pool_size = 2
kernel_size = 3

#%%
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def train_model(model, train, test, num_classes):
    x_train = train[0].reshape((train[0].shape[0],) + input_shape)
    x_test = test[0].reshape((test[0].shape[0],) + input_shape)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(train[1], num_classes)
    y_test = keras.utils.to_categorical(test[1], num_classes)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    t = now()
    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return history

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
x_train_lt5 = x_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
x_test_lt5 = x_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

feature_layers = [
    Conv2D(filters, kernel_size,
           padding='valid',
           input_shape=input_shape),
    Activation('relu'),
    Conv2D(filters, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=pool_size),
    Dropout(0.25),
    Flatten(),
]

classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
]


#%%
# create complete model
model = Sequential(feature_layers + classification_layers)

# train model for 5-digit classification [0..4]
train_model(model,
            (x_train_lt5, y_train_lt5),
            (x_test_lt5, y_test_lt5), num_classes)


#%%
x_train_gte5 = x_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5
x_test_gte5 = x_test[y_test >= 5]
y_test_gte5 = y_test[y_test >= 5] - 5

#%%
# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

# transfer: train dense layers for new classification task [5..9]
train_model(model,
            (x_train_gte5, y_train_gte5),
            (x_test_gte5, y_test_gte5), num_classes)

#%%
#images = glob.glob('train/*')
from PIL import Image 
import numpy as np 
import PIL.ImageOps
images = []
letters = ['A','B','C','D','E']
#Max of 24
examples = 6
test_examples = 4
size = (28,28)

def importImagesAsArrays(folder, letters, examples, size, ending = '.png'):
    images = []
    target = []
    for letter in letters:
        for i in range(1,examples+1):
            name = folder+'/'+letter+'_'+str(i)+ending
            print(name)
            im = Image.open(name)
            im = im.convert("L").resize(size)
            #im = PIL.ImageOps.invert(im)
            #plt.imshow(im)
            im = np.array(im)

            images.append(im)
            target.append(letters.index(letter))
    return (images, target)
images, abc_train_y = importImagesAsArrays('train',letters, examples, size)
test_images, abc_test_y = importImagesAsArrays('test',letters, test_examples, size ,'.jpg')

#%%
#train
np_abc_train_y = np.array(abc_train_y)
x_train = np.stack(images)
np_abc_train_x = x_train.reshape((len(letters) * examples,28,28,1))
#test
np_abc_test_y = np.array(abc_test_y)
x_test = np.stack(test_images)
#breaks based on image
np_abc_test_x = x_test.reshape((len(letters) * test_examples,28,28,1))

#%%
# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False
# transfer: train dense layers for new classification task [5..9]
history = train_model(model,
            (np_abc_train_x, np_abc_train_y), #train
            (np_abc_test_x, np_abc_test_y), #test
            num_classes)

#%%

def plot_training_curves(history, title=None):
    ''' Plot the training curves for loss and accuracy given a model history'''
    # find the minimum loss epoch
    minimum = np.min(history.history['val_loss'])
    min_loc = np.where(minimum == history.history['val_loss'])[0]
    # get the vline y-min and y-max
    loss_min, loss_max = (min(history.history['val_loss'] + history.history['loss']),
                          max(history.history['val_loss'] + history.history['loss']))
    acc_min, acc_max = (min(history.history['val_accuracy'] + history.history['accuracy']),
                        max(history.history['val_accuracy'] + history.history['accuracy']))
    # create figure
    fig, ax = plt.subplots(ncols=2, figsize = (15,7))
    fig.suptitle(title)
    index = np.arange(1, len(history.history['accuracy']) + 1)
    # plot the loss and validation loss
    ax[0].plot(index, history.history['loss'], label = 'loss')
    ax[0].plot(index, history.history['val_loss'], label = 'val_loss')
    ax[0].vlines(min_loc + 1, loss_min, loss_max, label = 'min_loss_location')
    ax[0].set_title('Loss')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].legend()
    # plot the accuracy and validation accuracy
    ax[1].plot(index, history.history['accuracy'], label = 'accuracy')
    ax[1].plot(index, history.history['val_accuracy'], label = 'val_accuracy')
    ax[1].vlines(min_loc + 1, acc_min, acc_max, label = 'min_loss_location')
    ax[1].set_title('Accuracy')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].legend()
    plt.show()
plot_training_curves(history)
# %%
