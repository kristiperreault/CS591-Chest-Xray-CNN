import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from tensorflow.keras import datasets, layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
from PIL import Image

import pathlib
import pickle

def augment_data(image):
    scale = random.random() + 0.5
    aug_image = tf.math.scalar_mul(scale, image)
    return aug_image

# Download and extract data
def load_transform_data():

    # Define Data Directories
    data_dir = pathlib.Path('chest_xray')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    val_dir = data_dir / 'val'

    # Create TF Datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, color_mode='grayscale', image_size=(150, 150))
    aug_ds = train_ds.map(lambda x, y: (augment_data(x), y))
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(5000, reshuffle_each_iteration=True)
    aug_ds = train_ds.cache()
    aug_ds = train_ds.shuffle(5000, reshuffle_each_iteration=True)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, color_mode='grayscale', batch_size=16, image_size=(150, 150))
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, color_mode='grayscale', batch_size=624, image_size=(150, 150))

    return train_ds, aug_ds, val_ds, test_ds

# Create the model
def create_model(dropout=False, contrast=False):

    # Create convolutional base with max pooling
    model = models.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1./255))
    if (contrast):
        model.add(layers.experimental.preprocessing.RandomContrast(0.5))
    if (dropout):
        model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2, 2)))
    if (dropout):
        model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2, 2)))
    if (dropout):
        model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.AveragePooling2D((2, 2)))

    # Add dense layers
    model.add(layers.Flatten())
    if (dropout):
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    #if (dropout):
    #    model.add(layers.Dropout(0.5))
    #model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2)) # This is our # of classes - one for pneumonia, one not

    return model


# Compile and train model
def train(model, train_ds, val_ds):

    # Compile the model; we can change optimizer type and metrics reported
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model - vary epochs
    history = model.fit(train_ds, epochs=25, validation_data=val_ds)
    
    return history


# Evaluate the model
def evaluate(model, history, test_ds, model_name):

    for test_images, test_labels in test_ds.as_numpy_iterator(): 
        # Precision and Recall
        predictions = np.argmax(model.predict(test_images), axis=-1)
        # precision tp / (tp + fp)
        precision = precision_score(test_labels, predictions)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(test_labels, predictions)
        print('Recall: %f' % recall)
        pickle.dump(predictions, open("models/{}/predictions.p".format(model_name), 'wb'))
        pickle.dump(test_labels, open("models/{}/test_labels.p".format(model_name), 'wb'))

    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print("Loss: {}".format(test_loss))
    print("Accuracy: {}".format(test_acc))

    # Plot ROC and AUC - from dataset code
#    y_test, y_labels = test_labels, predictions # Figure this out
#    pos_label = 1 # Figure this out
#    fpr, tpr, _ = roc_curve(y_test, y_labels, pos_label = pos_label)
#    roc_auc = auc(fpr, tpr)
#    plt.figure()
#    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
#    plt.plot([0, 1], [0, 1], "k--")
#    plt.xlim([0.0, 1.05])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel("False Positive Rate")
#    plt.ylabel("True Positive Rate")
#    plt.title("Receiver operating characteristic curve")
#    plt.show(block=True)

# Run CNN
train_ds, train_aug_ds, val_ds, test_ds = load_transform_data()

model_name = 'default'
print("Model: ", model_name)
model = create_model()
history = train(model, train_ds, val_ds)
evaluate(model, history, test_ds, model_name)
pickle.dump(history.history, open("models/{}/history.p".format(model_name), 'wb'))
model.summary()

model_name = 'dropout'
print("Model: ", model_name)
model = create_model(dropout=True)
history = train(model, train_ds, val_ds)
pickle.dump(history.history, open("models/{}/history.p".format(model_name), 'wb'))
evaluate(model, history, test_ds, model_name)
model.summary()

model_name = 'contrast'
print("Model: ", model_name)
model = create_model(contrast=True)
history = train(model, train_ds, val_ds)
pickle.dump(history.history, open("models/{}/history.p".format(model_name), 'wb'))
evaluate(model, history, test_ds, model_name)
model.summary()

model_name = 'both'
print("Model: ", model_name)
model = create_model(contrast=True, dropout=True)
history = train(model, train_ds, val_ds)
pickle.dump(history.history, open("models/{}/history.p".format(model_name), 'wb'))
evaluate(model, history, test_ds, model_name)
model.summary()

model_name = 'brightness'
print("Model: ", model_name)
model = create_model()
history = train(model, train_aug_ds, val_ds)
pickle.dump(history.history, open("models/{}/history.p".format(model_name), 'wb'))
evaluate(model, history, test_ds, model_name)
model.summary()

