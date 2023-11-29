# Load the packages
library(farff)
library(OpenML)
library(randomForest)
library(dplyr)
library(ggplot2)
library(caret)

# Load the Fashion MNIST dataset
fashion_mnist <- getOMLDataSet(data.id = 40996)

# Extract and prepare data
X <- as.data.frame(fashion_mnist$data)
# Check the names of the columns to identify the target column
print(names(X))

# Assuming 'class' is the name of the target column
y <- as.factor(X$class)  
X <- X[, names(X) != "class"]  # Remove the label column from X

# Split dataset into training and testing
set.seed(42)
train_indices <- sample(1:nrow(X), 0.7 * nrow(X))
X_train <- X[train_indices, ]
y_train <- y[train_indices]
X_test <- X[-train_indices, ]
y_test <- y[-train_indices]

# Train the Random Forest model
start_time <- Sys.time()
rf_model <- randomForest(x = X_train, y = y_train, ntree = 100, mtry = 28, do.trace = 50, njobs = 10)
end_time <- Sys.time()
training_time <- end_time - start_time
print(paste("Training time: ", training_time))

# Predict on the test set
predictions <- predict(rf_model, X_test)

# Calculate accuracy and other metrics
accuracy <- mean(predictions == y_test)
print(paste("Accuracy:", accuracy))
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
print(confusion_matrix)
classification_report <- confusionMatrix(predictions, y_test)
print(classification_report)

