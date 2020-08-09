import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

import pathlib

# Download and extract data
def load_transform_data():

    # Define Data Directories
    data_dir = pathlib.Path('chest_xray/chest_xray')
    train_dir = data_dir / 'train'
    test_dir = data_dir / 'test'
    val_dir = data_dir / 'val'

    # Create TF Datasets
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir, color_mode='grayscale')
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(val_dir, color_mode='grayscale')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir, color_mode='grayscale')

    return train_ds, val_ds, test_ds

# Create the model
def create_model():

    # Create convolutional base with max pooling
    # MaxPooling2D(poolsize, strides=, padding=, data_format)
    # Conv2D(filters, kernel_size, strides=, padding=, activation=, use_bias=, ..)
    # We can change these params, change pool size
    # Need to update image size
    model = models.Sequential()
    model.add(layers.experimental.preprocessing.Rescaling(1./255))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # Add layers
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2)) # This is our # of classes - one for pneumonia, one not

    return model


# Compile and train model
def train(model, train_images, val_images, test_images):

    # Compile the model; we can change optimizer type and metrics reported
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model - vary epochs
    history = model.fit(train_images, epochs=1, validation_data=val_images)
    
    return history


# Evaluate the model
def evaluate(model, history, test_images):

    test_images_data, y_labels = test_images

    # Precision and Recall
    y_classes = model.predict_classes(test_images_data, verbose=0)
    # precision tp / (tp + fp)
    precision = precision_score(test_images_data, y_classes)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(test_images_data, y_classes)
    print('Recall: %f' % recall)

    # Plot epoch vs accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, verbose=2)
    print("Loss: {}", test_loss)
    print("Accuracy: {}", test_acc)

    # Plot ROC and AUC - from dataset code
    pos_label = 1
    fpr, tpr, _ = roc_curve(test_images_data, y_labels, pos_label = pos_label)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic curve")
    plt.show()


# Run CNN
train_images, val_images, test_images = load_transform_data()
model = create_model()
history = train(model, train_images, val_images, test_images)
evaluate(model, history, test_images)
