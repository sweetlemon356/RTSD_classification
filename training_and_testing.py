from imgaug import augmenters as iaa
import random
import csv
import numpy as np
from PIL import Image
import tensorflow
#from keras.preprocessing import img_to_array
from sklearn.model_selection import train_test_split
#from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras import backend as K
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D




def cnt_img_in_classes(labels):
    print('alive')
    count = {}
    for i in labels:
        if i in count:
            count[i] += 1
        else:
            count[i] = 1
    print('end')
    return count


def diagram(count_classes):
    plt.bar(range(len(count_classes)), sorted(list(count_classes.values())), align='center')
    plt.xticks(range(len(count_classes)), sorted(list(count_classes.keys())), rotation=90, fontsize=7)
    plt.show()


def aug_images(images, p):
    augs = iaa.SomeOf((2, 4),
                      [
                          iaa.Crop(px=(0, 4)),
                          iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                          iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                          iaa.Affine(rotate=(-45, 45)),
                          iaa.Affine(shear=(-10, 10))
                      ])

    seq = iaa.Sequential([iaa.Sometimes(p, augs)])
    res = seq.augment_images(images)
    return res


def augmentation(images, labels):
    min_imgs = 100
    classes = cnt_img_in_classes(labels)
    for i in range(len(classes)):
        if (classes[i] < min_imgs):
            add_num = min_imgs - classes[i]
            imgs_for_augm = []
            lbls_for_augm = []
            for j in range(add_num):
                im_index = random.choice(np.where(labels == i)[0])
                imgs_for_augm.append(images[im_index])
                lbls_for_augm.append(labels[im_index])
            augmented_class = aug_images(imgs_for_augm, 1)
            augmented_class_np = np.array(augmented_class)
            augmented_lbls_np = np.array(lbls_for_augm)
            images = np.concatenate((images, augmented_class_np), axis=0)
            labels = np.concatenate((labels, augmented_lbls_np), axis=0)
    return (images, labels)


class Net:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
        model = Sequential()
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=inputShape))
        model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(rate=0.25))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dropout(rate=0.5))
        model.add(Dense(classes, activation='softmax'))
        return model


filename = 'gt_train.csv'

data = []
labels = []

with open('gt_train.csv') as file:
    file_reader = csv.reader(file, delimiter=",")
    count = 0
    for row in file_reader:
        if count != 0:
            img_path, class_ = row[0], int(row[1])
            img = Image.open(f"train/{img_path}").resize((30, 30))
            data.append((tensorflow.keras.utils.img_to_array(img)))
            labels.append(class_)
        count += 1
data = np.array(data)
labels = np.array(labels)


samples_distribution = cnt_img_in_classes(labels)
print(data.shape, labels.shape)
data, labels = augmentation(data, labels)
print(data.shape, labels.shape)
augmented_samples_distribution = cnt_img_in_classes(labels)
diagram(augmented_samples_distribution)

train_x, test_x, train_y, test_y = train_test_split(data, labels, test_size=0.2, random_state=67)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)

train_y = tensorflow.keras.utils.to_categorical(train_y, 67)
test_y = tensorflow.keras.utils.to_categorical(test_y, 67)

epochs = 25
sgd = tensorflow.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model = Net.build(width=30, height=30, depth=3, classes=67)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_x, train_y, batch_size=64, validation_data=(test_x, test_y), epochs=epochs)



plt.figure()
N = epochs
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()










