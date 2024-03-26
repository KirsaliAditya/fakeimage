import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

file_path = r"E:\imagefake\CASIA2"

# Using tf.keras.preprocessing.image_dataset_from_directory to load the dataset
batch_size = 32
image_size = (32, 32)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)

train_dataset = train_datagen.flow_from_directory(
    file_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='sparse',  # Since you are using sparse categorical crossentropy
    subset="training",
    seed=1337,
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    file_path,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# Building our stronger model
strong_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu',padding='same'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu',padding='same'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the stronger model
strong_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for the stronger model
checkpoint_strong = ModelCheckpoint("best_strong_model.h5", monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
early_stopping_strong = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training the stronger model using the loaded datasets
history_strong = strong_model.fit(
    train_dataset,
    epochs=20,  # Increase the number of epochs for better convergence
    validation_data=validation_dataset,
    callbacks=[checkpoint_strong, early_stopping_strong]
)

# Evaluate the stronger model on the test set
loss_strong, accuracy_strong = strong_model.evaluate(validation_dataset)
print('Accuracy of the stronger model:', accuracy_strong)

# Assuming you have validation labels and predictions
validation_labels_strong = []
validation_predictions_strong = []

for x, y in validation_dataset:
    validation_labels_strong.extend(y.numpy())  # Labels in one-hot format
    validation_predictions_strong.extend(np.argmax(strong_model.predict(x), axis=1))

# Compute confusion matrix for the stronger model
conf_matrix_strong = confusion_matrix(validation_labels_strong, validation_predictions_strong)

# Extract TP, FP, TN, FN from the confusion matrix for the stronger model
tp_strong = conf_matrix_strong[1, 1]  # True Positive
fp_strong = conf_matrix_strong[0, 1]  # False Positive
tn_strong = conf_matrix_strong[0, 0]  # True Negative
fn_strong = conf_matrix_strong[1, 0]  # False Negative

# Calculate precision, recall, and F1 score for the stronger model
precision_strong = tp_strong / (tp_strong + fp_strong)
recall_strong = tp_strong / (tp_strong + fn_strong)
f1_score_strong = 2 * (precision_strong * recall_strong) / (precision_strong + recall_strong)

print(f'Precision of the stronger model: {precision_strong:.4f}')
print(f'Recall of the stronger model: {recall_strong:.4f}')
print(f'F1 Score of the stronger model: {f1_score_strong:.4f}')

# Alternatively, you can use the classification_report function for the stronger model
print('\nClassification Report for the stronger model:')
print(classification_report(validation_labels_strong, validation_predictions_strong,zero_division=1))

# Save the stronger model
strong_model.save('my_strong_model.h5')