import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
import random
import PIL
from PIL import Image

import pathlib
import pickle

# Download and extract data
def load_transform_data():

    # Define Data Directories
    data_dir = pathlib.Path('chest_xray')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    val_dir = data_dir / 'val'

    # Create TF Datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, color_mode='grayscale')
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(5000, reshuffle_each_iteration=True)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, color_mode='grayscale', batch_size=16)
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, color_mode='grayscale', batch_size=624)

    return train_ds, val_ds, test_ds

# Create the model
def create_model(dropout=False, contrast=False):

    # Create convolutional base with max pooling
    # MaxPooling2D(poolsize, strides=, padding=, data_format)
    # Conv2D(filters, kernel_size, strides=, padding=, activation=, use_bias=, ..)
    model = models.Sequential()
    if (contrast):
        model.add(layers.experimental.preprocessing.RandomContrast((0.5, 0.5)))
    model.add(layers.experimental.preprocessing.Rescaling(1./255))
    if (dropout):
        model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    if (dropout):
        model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    if (dropout):
        model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Add dense layers
    model.add(layers.Flatten())
    if (dropout):
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    if (dropout):
        model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2)) # This is our # of classes - one for pneumonia, one not

    return model


# Compile and train model
def train(model, train_ds, val_ds):

    # Compile the model; we can change optimizer type and metrics reported
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model - vary epochs
    history = model.fit(train_ds, epochs=50, validation_data=val_ds)
    
    return history


# Evaluate the model
def evaluate(model, history, test_ds, model_name):

    for test_images, test_labels in test_ds.as_numpy_iterator(): 
        # Precision and Recall
        predictions = np.argmax(model.predict(test_images), axis=-1)
        pickle.dump(predictions, open("models/{}/predictions.p".format(model_name), 'wb'))
        # precision tp / (tp + fp)
        precision = precision_score(test_labels, predictions)
        print('Precision: %f' % precision)
        # recall: tp / (tp + fn)
        recall = recall_score(test_labels, predictions)
        print('Recall: %f' % recall)

    # Plot epoch vs accuracy
#    plt.plot(history['accuracy'], label='accuracy')
#    plt.plot(history['val_accuracy'], label = 'val_accuracy')
#    plt.xlabel('Epoch')
#    plt.ylabel('Accuracy')
#    plt.ylim([0.5, 1])
#    plt.legend(loc='lower right')
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print("Loss: {}".format(test_loss))
    print("Accuracy: {}".format(test_acc))

    # Plot ROC and AUC - from dataset code
#    y_test, y_labels = 0, 0 # Figure this out
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
#    plt.show()

# Run CNN
train_ds, val_ds, test_ds = load_transform_data()

#model_name = 'default'
#print("Model: ", model_name)
#model = create_model()
#history = train(model, train_ds, val_ds)
#evaluate(model, history, test_ds, model_name)
#pickle.dump(history.history, open("models/{}/history.p".format(model_name), 'wb'))
#model.summary()

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
