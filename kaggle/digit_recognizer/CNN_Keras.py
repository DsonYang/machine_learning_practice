#https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6

from keras.callbacks.callbacks import History
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage.measurements import label
import seaborn as sns
from sklearn import metrics
from tensorflow.python.keras.backend_config import epsilon
from tensorflow.python.training import optimizer

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

sns.set(style='white', context='notebook', palette='deep')


train = pd.read_csv(r'dataset/kaggle/digit_recognizer/train.csv')
test = pd.read_csv(r'dataset/kaggle/digit_recognizer/test.csv')

Y_train = train['label']
X_train = train.drop(labels=['label'], axis=1)
del train

# plt.figure(figsize=(10, 7))
# sns.countplot(Y_train)
# plt.savefig("output_figures/kaggle/digit_recognizer/Y_Train_Count.pdf")
# print(Y_train.value_counts())

#Check for null and missing valuse
print(X_train.isnull().any().describe())
print(test.isnull().any().describe())

#Normalization
X_train = X_train / 255.0
test = test / 255.0
#Reshape image in 3 dimensions (height=28, width=28,canal=1)
X_train = X_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors
Y_train = to_categorical(Y_train, num_classes=10)

#Split training and valdiation set
random_seed = 2
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1, random_state=random_seed)

# plt.figure(figsize=(10, 7))
# plt.imshow(X_train[15][:,:,0])
# plt.savefig("output_figures/kaggle/digit_recognizer/Data_img.pdf")

def Build_CNN_Model():
    # In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out
    theModel = Sequential()
    theModel.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28,28,1)))
    theModel.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))
    theModel.add(MaxPool2D(pool_size=(2,2)))
    theModel.add(Dropout(0.25))

    theModel.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
    theModel.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))
    theModel.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    theModel.add(Dropout(0.25))

    theModel.add(Flatten())
    theModel.add(Dense(256, activation='relu'))
    theModel.add(Dropout(0.5))
    theModel.add(Dense(10,activation='softmax'))

    #Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    theModel.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['acc'])

    return theModel




def EvaluateTheModel():
    #Data augmentation to avoid overfit
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        # randomly rotate images in the range(degrees, 0-100)
        rotation_range=10,
        zoom_range=0.1,  # randomly zoom image
        # randomly shift images horizontally(fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically(fraction of total height)
        height_shift_range=0.1,
        horizontal_flip=False,  # randomly flip images horizontally
        vertical_flip=False  # randomly flip images vertically
    )
    datagen.fit(X_train)
    epochs = 10
    batch_size = 86
    #reduce the learn_rate by half if the accuracy is not improved after 3 epochs, to avoid falling into local minima
    learning_rate_reduction = ReduceLROnPlateau(
        monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
    history = m.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        epochs=epochs, validation_data=(X_val, Y_val),
        verbose=2, steps_per_epoch=X_train.shape[0]//batch_size,
        callbacks=[learning_rate_reduction]
    )

    plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots(2 ,1)
    ax[0].plot(history.history['loss'], color='b', label='Training loss')
    ax[0].plot(history.history['val_loss'], color='r', label='validation loss', axes=ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label='Training accuracy')
    ax[1].plot(history.history['val_acc'], color='r',
               label='validation accuracy')
    legend = ax[1].legend(loc='best', shadow=True)
    plt.savefig("output_figures/kaggle/digit_recognizer/loss_accuracy.pdf")


def Plot_Confusion_Matrix(name):
    Y_pred = m.predict(X_val)
    Y_pred_classes = np.argmax(Y_pred, axis=1)
    Y_true = np.argmax(Y_val, axis=1)
    cm = confusion_matrix(Y_true, Y_pred_classes)

    plt.figure(figsize=(10, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    classes = range(10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, cm[i, j],
            horizontalalignment = 'center',
            color = 'white' if cm[i,j] > thresh else 'black'
        )
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicetd label')
    plt.savefig("output_figures/kaggle/digit_recognizer/"+name+".pdf")

m = Build_CNN_Model()
Plot_Confusion_Matrix('confusion_matrix_without_train')
EvaluateTheModel()
Plot_Confusion_Matrix('confusion_matrix_after_train')
    
