import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="train/display")
mode = ap.parse_args().mode

# Plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    # Print available keys in history
    history_dict = model_history.history
    print("Keys in model history:", history_dict.keys())

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Handle both 'accuracy' and 'acc'
    if 'accuracy' in history_dict:
        train_acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
    elif 'acc' in history_dict:
        train_acc = history_dict['acc']
        val_acc = history_dict['val_acc']
    else:
        print("Accuracy key not found in history.")
        return

    # Summarize history for accuracy
    epochs = np.arange(1, len(train_acc) + 1)
    
    axs[0].plot(epochs, train_acc, label='Train Accuracy')
    axs[0].plot(epochs, val_acc, label='Val Accuracy')
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(epochs)
    axs[0].legend(loc='best')
    
    # Summarize history for loss
    axs[1].plot(epochs, history_dict['loss'], label='Train Loss')
    axs[1].plot(epochs, history_dict['val_loss'], label='Val Loss')
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(epochs)
    axs[1].legend(loc='best')
    
    # Save the figure
    fig.savefig('plot.png')
    plt.show()

# Define data generators
train_dir = 'data/train'
val_dir = 'data/test'

num_train = 28709
num_val = 7178
batch_size = 64
num_epoch = 180

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode='categorical')

# Create or load the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


if os.path.exists('model.h5'):
    print("Loading model from 'model.h5'.")
    model = load_model('model.h5')
    initial_epoch = 170  # Set to the number of epochs already completed
else:
    print("No existing model found. Creating a new one.")
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']
    )
    initial_epoch = 0  # Start from scratch

# Use ModelCheckpoint with the .h5 format
checkpoint = ModelCheckpoint('model.h5', save_best_only=False, save_format='h5')

if mode == "train":
    model_info = model.fit(
        train_generator,
        steps_per_epoch=num_train // batch_size,
        epochs=num_epoch,
        validation_data=validation_generator,
        validation_steps=num_val // batch_size,
        initial_epoch=initial_epoch,
        callbacks=[checkpoint]
    )
    
    plot_model_history(model_info)
    
    # Load previous history if it exists
    if os.path.exists('history.json'):
        with open('history.json', 'r') as f:
            old_history = json.load(f)
    else:
        old_history = {}

    # Function to append or create new history entries
    def merge_histories(old, new):
        for key in new:
            if key in old:
                old[key].extend(new[key])
            else:
                old[key] = new[key]
        return old

    # Merge new history with old
    full_history = merge_histories(old_history, model_info.history)

    # Save the full history back to the JSON file
    with open('history.json', 'w') as f:
        json.dump(full_history, f)

elif mode == "display":
    model.load_weights('model.h5')

    # Prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # Dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # Start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 169, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
