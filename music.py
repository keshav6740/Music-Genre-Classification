import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from scipy.signal import spectrogram
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']
SAMPLE_RATE = 22050
TIME_FRAMES = 250  
N_MELS = 128  

def load_and_preprocess_data(data_dir):
    features, labels = [], []
    
    for genre_idx, genre in enumerate(GENRES):
        genre_path = os.path.join(data_dir, genre)
        for audio_file in os.listdir(genre_path):
            if audio_file.endswith('.wav'):
                audio_path = os.path.join(genre_path, audio_file)
                
                try:
                    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                    
                    mel_spec = librosa.feature.melspectrogram(
                        y=y, 
                        sr=sr, 
                        n_mels=N_MELS, 
                        n_fft=2048, 
                        hop_length=512
                    )
                    
                    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                    
                    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
                    
                    if mel_spec_db.shape[1] > TIME_FRAMES:
                        mel_spec_db = mel_spec_db[:, :TIME_FRAMES]
                    else:
                        pad_width = TIME_FRAMES - mel_spec_db.shape[1]
                        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
                    
                    features.append(mel_spec_db)
                    labels.append(genre_idx)
                
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
    
    return np.array(features)[..., np.newaxis], keras.utils.to_categorical(labels)

def create_enhanced_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), padding='same', activation='relu', 
                      kernel_regularizer=l2(0.001), input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.SpatialDropout2D(0.2),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu', 
                      kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.SpatialDropout2D(0.2),
        
        layers.Conv2D(256, (3, 3), padding='same', activation='relu', 
                      kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.SpatialDropout2D(0.2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

data_dir = r'D:/Projects/DeepLearning/Data/genres_original'
X, y = load_and_preprocess_data(data_dir)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

model = create_enhanced_model(X_train.shape[1:], len(GENRES))
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.2, 
    patience=5, 
    min_lr=0.00001
)

early_stop = EarlyStopping(
    monitor='val_accuracy', 
    patience=10, 
    restore_best_weights=True
)

# Training
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=100,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stop]
)

# Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Confusion Matrix and Classification Report
from sklearn.metrics import confusion_matrix, classification_report
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_true_classes, y_pred_classes))
print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=GENRES))

# Save Model
model.save('enhanced_music_genre_classifier.h5')