import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Download and extract data
def load_transform_data():

    # Load data

    # Perform any data manipulation

    return 0, 0, 0, 0

# Create the model
def create_model():

    # Create convolutional base with max pooling
    # MaxPooling2D(poolsize, strides=, padding=, data_format)
    # Conv2D(filters, kernel_size, strides=, padding=, activation=, use_bias=, ..)
    # We can change these params, change pool size
    # Need to update image size
    model = models.Sequential()
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
def train(model, train_images, train_labels, test_images, test_labels):

    # Compile the model; we can change optimizer type and metrics reported
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    # Train the model - vary epochs
    history = model.fit(train_images, train_labels, epochs=10, 
                        validation_data=(test_images, test_labels))
    
    return history


# Evaluate the model
def evaluate(model, history, test_images, test_labels):

    # Precision and Recall


    # Plot epoch vs accuracy
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print("Loss: {}", test_loss)
    print("Accuracy: {}", test_acc)

    # Plot ROC and AUC - from dataset code
    y_test, y_labels = 0, 0 # Figure this out
    pos_label = 1 # Figure this out
    fpr, tpr, _ = roc_curve(y_test, y_labels, pos_label = pos_label)
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
train_images, train_labels, test_images, test_labels = load_transform_data()
model = create_model()
history = train(model, train_images, train_labels, test_images, test_labels)
evaluate(model, history, test_images, test_labels)
