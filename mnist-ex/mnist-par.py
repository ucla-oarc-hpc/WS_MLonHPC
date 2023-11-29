from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import fetch_openml
import time

# Load the Fashion MNIST dataset from OpenML
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier with n_jobs=10
clf = RandomForestClassifier(random_state=42, n_jobs=10)

# Start the timer
start_time = time.time()

# Train the model
clf.fit(X_train, y_train)

# End the timer
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f"Training Time: {training_time} seconds")

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy}")

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
