import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set the path to the dataset
dataset_path = 'dogs-vs-cats/train'
# https://www.kaggle.com/c/dogs-vs-cats/data
# Load and preprocess the dataset
def load_images(dataset_path, img_size=(64, 64)):
    images = []
    labels = []
    for category in ['cat', 'dog']:
        path = os.path.join(dataset_path, category)
        class_num = 0 if category == 'cat' else 1
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load in grayscale
                img = cv2.resize(img, img_size)  # Resize image
                images.append(img)
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")
    return np.array(images), np.array(labels)

# Prepare the data
X, y = load_images(dataset_path)
X = X.reshape(X.shape[0], -1)  # Flatten images into vectors
X = X / 255.0  # Normalize pixel values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the SVM model
model = SVC(kernel='linear')  # You can also try 'rbf', 'poly', etc.
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Detailed classification report
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Make a prediction on a new image
def predict_image(model, img_path, img_size=(64, 64)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img.flatten().reshape(1, -1)
    img = img / 255.0
    prediction = model.predict(img)
    return 'Dog' if prediction[0] == 1 else 'Cat'

# Example prediction
example_img_path = 'dogs-vs-cats/test/cat/1234.jpg'  # Replace with a valid path
prediction = predict_image(model, example_img_path)
print(f'Predicted Label: {prediction}')
