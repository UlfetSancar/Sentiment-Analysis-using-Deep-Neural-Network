import os
import logging
import time
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, Bidirectional # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import numpy as np

# Set environment variables
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")# Suppress warnings from Python itself

# Define the path to CSV file
file_path = "Combined_Data.csv"

# Check if the file exists
if os.path.exists(file_path):
    # Load the data
    data = pd.read_csv(file_path)
    print("File loaded successfully.")

    # Display columns and first few rows
    print("Columns in DataFrame:", data.columns)
    print(data.head())

    # Handle missing values
    data['statement'] = data['statement'].fillna('')

    # Extract text and labels
    X = data['statement']
    y = data['status']

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tokenize the text data
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Pad the sequences
    max_length = 100
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

    # Define the DNN model
    model = Sequential([
        Embedding(input_dim=10000, output_dim=64),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.5),
        Bidirectional(LSTM(32)),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    start_time = time.time()

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))

    # Train the model with class weights
    history = model.fit(X_train_pad, y_train, epochs=10, batch_size=32, validation_split=0.2)
    training_time = time.time() - start_time

    # Evaluate the model
    start_time = time.time()
    test_loss, test_accuracy = model.evaluate(X_test_pad, y_test)
    evaluation_time = time.time() - start_time

    print(f"\nTraining Time: {training_time:.2f} seconds")
    print(f"Evaluation Time: {evaluation_time:.2f} seconds")

    # Predict on the test set
    y_pred_probs = model.predict(X_test_pad, verbose=0)
    y_pred = y_pred_probs.argmax(axis=-1)

    # Decode predictions and true labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    # Classification report
    print("\nClassification Report:\n", classification_report(y_test_decoded, y_pred_decoded, zero_division=0))


    # Confusion matrix
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate F1-scores
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    print(f"\nMacro-average F1-score: {f1_macro:.2f}")
    print(f"Weighted F1-score: {f1_weighted:.2f}\n")  # Adds a blank line after Weighted F1-score


    # Visualize F1-scores per class
    f1_per_class = f1_score(y_test, y_pred, average=None)
    plt.figure(figsize=(10, 6))
    plt.bar(label_encoder.classes_, f1_per_class)
    plt.title("F1-scores per Class")
    plt.xlabel("Class")
    plt.ylabel("F1-score")
    plt.xticks(rotation=45)
    plt.show()

    # Analyze class distribution in predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    clean_distribution = {int(k): int(v) for k, v in dict(zip(unique, counts)).items()}
    print("\nPredicted class distribution:", clean_distribution)

    # Predict on the test set
    y_pred_probs = model.predict(X_test_pad)
    y_pred = y_pred_probs.argmax(axis=-1)

    # Decode predictions and true labels
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)

    # Analyze class distribution in predictions
    unique, counts = np.unique(y_pred, return_counts=True)
    print("Predicted class distribution:", dict(zip(unique, counts)))

    # BASELINE: Random Guessing
    unique_classes = np.unique(y_train)
    random_predictions = np.random.choice(unique_classes, size=len(y_test), replace=True)

    random_accuracy = accuracy_score(y_test, random_predictions)
    print(f"\nRandom Guessing Accuracy: {random_accuracy * 100:.2f}%")
    print("\nRandom Guessing Classification Report:\n",
          classification_report(y_test, random_predictions, target_names=label_encoder.classes_))

    # BASELINE: Most Frequent Class
    most_frequent_class = np.argmax(np.bincount(y_train))
    most_frequent_predictions = np.full_like(y_test, most_frequent_class)

    most_frequent_accuracy = accuracy_score(y_test, most_frequent_predictions)
    print(f"Most Frequent Class Accuracy: {most_frequent_accuracy * 100:.2f}%")
    print("\nMost Frequent Class Classification Report:\n",
          classification_report(y_test, most_frequent_predictions, target_names=label_encoder.classes_))

    # Generate confusion matrices for baselines
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes = axes.flatten()

    # Random Guessing Confusion Matrix
    cm_random = confusion_matrix(y_test, random_predictions)
    sns.heatmap(cm_random, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=axes[0])
    axes[0].set_title("Random Guessing Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Most Frequent Class Confusion Matrix
    cm_most_frequent = confusion_matrix(y_test, most_frequent_predictions)
    sns.heatmap(cm_most_frequent, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_, ax=axes[1])
    axes[1].set_title("Most Frequent Class Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()

else:
    print(f"File not found: {file_path}")
