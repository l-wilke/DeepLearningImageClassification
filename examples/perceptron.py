
# A very simple perceptron for classifying american sign language letters
import signdata
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils
import wandb
from wandb.keras import WandbCallback
from keras.applications import ResNet50, VGG16
from keras import optimizers

from keras import layers

seed = 7
np.random.seed(seed)

# logging code
run = wandb.init()

if (config.team_name == 'default'):
    raise ValueError("Please set config.team_name to be your team name")

# load data
(X_test, y_test) = signdata.load_test_data()
(X_train, y_train) = signdata.load_train_data()

print (X_test.shape)

img_width = X_test.shape[1]

image_size = 197
#image_size = img_width
print (img_width)
print (img_height)

#resnet = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

for layer in resnet.layers[:-4]:
    layer.trainable = False

for layer in resnet.layers:
    print (layer, layer.trainable)


def resize(res):
    base_size = 197
    height, width = res.shape

    if height < base_size:
        diff = base_size - height
        extend_top = diff // 2
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, extend_top, extend_bottom, 0, 0, borderType=cv2.BORDER_CONSTANT, value=0)

    if width < base_size:
        diff = base_size - width
        extend_top = diff // 2
        extend_bottom = diff - extend_top
        res = cv2.copyMakeBorder(res, 0, 0, extend_top, extend_bottom, borderType=cv2.BORDER_CONSTANT, value=0)

    return res

print ('doing train resize')
for i in X_train:
    res = resize(i)
    trains.append(np.stack((res,)*3, -1))

print ('after appending')
#X_itrain = np.expand_dims(numpy_image, axis=0)

X_itrain = np.array(trains)
print ('new', X_itrain.shape)

tests = []

print ('doing test resize')
for i in X_test:
    res = resize(i)
    tests.append(np.stack((res,)*3, -1))

X_itest = np.array(tests)

print ('new2', X_itest.shape)
#numpy_image = img_to_array(X_test)
#X_itest = np.expand_dims(numpy_image, axis=0)

#model.fit(X_train, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
#model.fit(X_itrain, y_train, epochs=config.epochs, validation_data=(X_test, y_test),
print ('doing fit')
model.fit(X_itrain, y_train, epochs=config.epochs, validation_data=(X_itest, y_test), shuffle=True, batch_size=128)
