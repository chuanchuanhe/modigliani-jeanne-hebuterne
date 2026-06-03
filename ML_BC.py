import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold


# Set seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)


# Load and preprocess the breast cancer wisconsin dataset
def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1, errors='ignore').values

    # M(malignant) = 1, B(benign) = 0
    y = np.where(df['diagnosis'] == 'M', 1, 0)

    return X, y

X, y = preprocess_data('C:/Users/chuan/Downloads/data.csv')


# Standardize features using z-score normalization
def standardize_train_test(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    return X_train_scaled, X_test_scaled


# Implement K-Nearest Neighbors (KNN) from scratch
class KNN:
    def __init__(self, k_neighbors=5):
        # Number of nearest neighbors to consider
        self.k = k_neighbors

    def fit(self, X, y):
        # Store training data
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        # Predict label for each test sample
        predictions = []

        for sample in X_test:
            predicted_label = self._predict_single(sample)
            predictions.append(predicted_label)

        return np.array(predictions)

    def _predict_single(self, sample):
        # Compute distance from this sample to ALL training points
        distances = []

        for train_sample in self.X_train:
            distance = np.sqrt(np.sum((sample - train_sample) ** 2))
            distances.append(distance)

        # Get indices that would sort the distances (smallest first)
        sorted_indices = np.argsort(distances)

        # Select the k closest neighbors
        nearest_indices = sorted_indices[:self.k]
    
        # Get their labels
        nearest_labels = [self.y_train[i] for i in nearest_indices]

        # Count how many times each label appears
        label_counts = {}

        for label in nearest_labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1

        # Find the label with the highest count
        # Start with no best label
        best_label = None
        highest_count = 0

        # Go through each label and its count
        for label, count in label_counts.items():
            if count > highest_count:
                highest_count = count
                best_label = label

        predicted_label = best_label
        
        return predicted_label


# Implement Logistic Regression from scratch
# probability = sigmoid(z)
class LogisticRegression:
    def __init__(self, lr, iterations):
        self.lr = lr
        self.iterations = iterations
        self.loss_history = []

    def sigmoid(self, z):
        # Keep z between -500 and 500 to prevent overflow in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Number of samples (m) and features (n)
        num_samples, num_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(num_features)
        self.bias = 0
        self.loss_history = []

        # Training loop (gradient descent)
        for iteration in range(self.iterations):

            # Compute linear combination (z = Xw + b)
            linear_output = np.dot(X, self.weights) + self.bias

            # Apply sigmoid to get probabilities
            predicted_probs = self.sigmoid(linear_output)

            # Compute binary cross-entropy loss
            loss = -np.mean(y * np.log(predicted_probs + 1e-9) + (1 - y) * np.log(1 - predicted_probs + 1e-9))

            # Store loss for plotting
            self.loss_history.append(loss)

            # Compute gradients for logistic regression
            error = predicted_probs - y
            gradient_weights = (1 / num_samples) * np.dot(X.T, error)
            gradient_bias = (1 / num_samples) * np.sum(error)

            # Update parameters and move in direction that reduces loss
            self.weights -= self.lr * gradient_weights
            self.bias -= self.lr * gradient_bias

        return self.loss_history

    # Default threshold is set to 0.5
    def predict(self, X, threshold=0.5):
        # Compute linear output
        linear_output = np.dot(X, self.weights) + self.bias

        # Convert to probabilities
        predicted_probs = self.sigmoid(linear_output)

        # Convert probabilities to class labels (0 or 1)
        predictions = []

        for prob in predicted_probs:
            if prob >= threshold:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)
    

# Define the PyTorch Neural Network architecture
class BreastCancerNet(nn.Module):
    def __init__(self, input_size=30, hidden_size=16, output_size=1):
        super(BreastCancerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out  # probability of class 1 (malignant)
    

def train_pytorch_nn(X_train, y_train, lr, epochs, batch_size):
    x_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    model = BreastCancerNet(input_size=X_train.shape[1])

    criterion = nn.BCELoss()   # BCELoss expects probabilities (0–1)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)  # optimizer using Stochastic Gradient Descent (SGD) with momentum

    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()    # reset gradients from previous batch
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward() # compute gradients (how each parameter contributed to the error)
            optimizer.step()  # update weights and bias using gradients
            epoch_loss += loss.item()   # accumulate batch loss to compute average loss for this epoch

        average_loss = epoch_loss / len(train_loader)  # compute average loss per batch for this epoch
        loss_history.append(average_loss)

    return model, loss_history


def cross_validate_pytorch_nn(X, y, folds=5, lr=0.01, epochs=100, batch_size=16):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    fold_metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }

    all_loss_histories = []

    splits = list(skf.split(X, y))

    # for fold_number, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    for fold_index in range(len(splits)):
        # Fold number starting from 1 (instead of 0)
        fold_number = fold_index + 1
        print(f"Training Neural Network Fold {fold_number}...")
        train_idx, test_idx = splits[fold_index]
        
        # Split raw data
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]

        # Normalize using training fold only to avoid leakage
        X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)

        # Train
        model, loss_history = train_pytorch_nn(
            X_train,
            y[train_idx],
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
        )

         # Predict on test fold
        preds = predict_pytorch_nn(model, X_test)

        results = calculate_metrics(y[test_idx], preds)

        fold_metrics["Accuracy"].append(results["Accuracy"])
        fold_metrics["Precision"].append(results["Precision"])
        fold_metrics["Recall"].append(results["Recall"])
        fold_metrics["F1-Score"].append(results["F1-Score"])

        all_loss_histories.append(loss_history)

    summary = {}

    for metric, values in fold_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        summary[metric] = {
            "Mean": mean_value,
            "Std": std_value
        }
    return summary, all_loss_histories


def predict_pytorch_nn(model, X_test):
    x_test = torch.FloatTensor(X_test)
    # Evaluation
    model.eval()
    with torch.no_grad():
        probabilities = model(x_test)
        # predictions = (probabilities >= 0.5).int().numpy().flatten()
        # Convert probabilities to True/False using threshold
        is_malignant = probabilities >= 0.5
        # Convert True/False to 1/0
        predictions_tensor = is_malignant.int()
        # Convert PyTorch tensor to NumPy array
        predictions_array = predictions_tensor.numpy()
        # Flatten to 1D array
        predictions = predictions_array.flatten()

    return predictions


# Evaluation Metrics 
def calculate_metrics(y_actual, y_pred):
    y_actual = np.array(y_actual)
    y_pred = np.array(y_pred)

    tp = np.sum((y_actual == 1) & (y_pred == 1))
    tn = np.sum((y_actual == 0) & (y_pred == 0))
    fp = np.sum((y_actual == 0) & (y_pred == 1))
    fn = np.sum((y_actual == 1) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_actual)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1_score,
        "Confusion Matrix": {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }
    }


# 5-Fold Cross Validation for KNN and Logistic Regression
# **kwargs allows flexible passing of model-specific parameters into the training function
def graduate_cross_validate(model_class, X, y, folds=5, **kwargs):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))

    fold_metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1-Score": []
    }

    for fold_index in range(len(splits)):
        fold_number = fold_index + 1
        print(f"Training Fold {fold_number}...")
        train_idx, test_idx = splits[fold_index]
        
        # Split the raw data
        X_train_raw = X[train_idx]
        X_test_raw = X[test_idx]

        # Normalize using train data only to prevent data leakage
        X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)

        # Train model on scaled data
        model = model_class(**kwargs)
        model.fit(X_train, y[train_idx])

        # Predict on scaled test
        preds = model.predict(X_test)

        results = calculate_metrics(y[test_idx], preds)

        fold_metrics["Accuracy"].append(results["Accuracy"])
        fold_metrics["Precision"].append(results["Precision"])
        fold_metrics["Recall"].append(results["Recall"])
        fold_metrics["F1-Score"].append(results["F1-Score"])

    summary = {}

    for metric, values in fold_metrics.items():
        mean_value = np.mean(values)
        std_value = np.std(values)
        summary[metric] = {
            "Mean": mean_value,
            "Std": std_value
        }

    return summary


# logistic regression learning rate tuning 
# randomly selecting 80% of the entire dataset as a training subset for hyperparameter (lr) tuning
indices = np.random.permutation(len(X))

train_size = int(0.8 * len(X))
train_idx = indices[:train_size]

X_train = X[train_idx]
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
y_train = y[train_idx]

learning_rates = [0.1, 0.01, 0.001]

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    model = LogisticRegression(lr=lr, iterations=1000)
    history = model.fit(X_train, y_train)

    plt.plot(history, label=f"lr={lr}")

plt.title("Logistic Regression: Impact of Learning Rate on Convergence")
plt.xlabel("Iterations")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()


# NN Learning Rate Tuning
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    model, history = train_pytorch_nn(
        X_train,
        y_train,
        lr=lr,
        epochs=100,
        batch_size=16
    )

    plt.plot(history, label=f"NN lr={lr}")

plt.title("Neural Network: Impact of Learning Rate on Convergence")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()


# NN batch size tuning
batch_sizes = [16, 32, 64]

plt.figure(figsize=(10, 6))

for batch_size in batch_sizes:
    model, history = train_pytorch_nn(
        X_train,
        y_train,
        lr=0.01,
        epochs=100,
        batch_size=batch_size
    )

    plt.plot(history, label=f"Batch Size={batch_size}")

plt.title("Neural Network: Mini-Batch Size Tuning")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()


# Run model comparison 
print("\nBenchmarking KNN...")
knn_results = graduate_cross_validate(
    KNN,
    X,
    y,
    folds=5,
    k_neighbors=5
)

print("\nBenchmarking Logistic Regression...")
lr_results = graduate_cross_validate(
    LogisticRegression,
    X,
    y,
    folds=5,
    lr=0.1,
    iterations=1500
)

print("\nBenchmarking PyTorch Neural Network...")
nn_results, nn_loss_histories = cross_validate_pytorch_nn(
    X,
    y,
    folds=5,
    lr=0.01,
    epochs=100,
    batch_size=16
)

# NN Learning Curves for 5-Fold Cross Validation
plt.figure(figsize=(10, 6))

for i in range(len(nn_loss_histories)):
    plt.plot(nn_loss_histories[i], label=f"Fold {i + 1}")

plt.title("Neural Network Learning Curves Across 5 Folds")
plt.xlabel("Epochs")
plt.ylabel("Binary Cross-Entropy Loss")
plt.legend()
plt.grid(True)
plt.show()


# Print results summary
all_results = [
    ("KNN", knn_results),
    ("Logistic Regression", lr_results),
    ("Neural Network", nn_results)
]

for model_name, results in all_results:
    print(f"\n--- {model_name} Results ---")

    for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        mean = results[metric]["Mean"]
        std = results[metric]["Std"]

        print(f"{metric}: {mean:.4f} (+/- {std:.4f})")


# Plot model comparison
model_names = ["KNN", "Logistic Regression (Threshold 0.5)", "Neural Network"]
results_list = [knn_results, lr_results, nn_results]

accuracy_means = [res["Accuracy"]["Mean"] for res in results_list]
precision_means = [res["Precision"]["Mean"] for res in results_list]
recall_means = [res["Recall"]["Mean"] for res in results_list]
f1_means = [res["F1-Score"]["Mean"] for res in results_list]

x_axis = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(x_axis - 1.5 * width, accuracy_means, width, label="Accuracy")
bar2 = ax.bar(x_axis - 0.5 * width, precision_means, width, label="Precision")
bar3 = ax.bar(x_axis + 0.5 * width, recall_means, width, label="Recall")
bar4 = ax.bar(x_axis + 1.5 * width, f1_means, width, label="F1-Score")

ax.set_ylabel("Score")
ax.set_title("Performance Comparison Using Stratified 5-Fold Cross Validation")
ax.set_xticks(x_axis)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)
ax.legend()

ax.bar_label(bar1, padding=3, fmt="%.3f")
ax.bar_label(bar2, padding=3, fmt="%.3f")
ax.bar_label(bar3, padding=3, fmt="%.3f")
ax.bar_label(bar4, padding=3, fmt="%.3f")

plt.tight_layout()
plt.show()


# Logistic Regression Threshold Comparison (0.5 vs 0.3)
print("\nComparing logistic regression thresholds 0.5 vs 0.3...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

recall_05 = []
recall_03 = []

precision_05 = []
precision_03 = []

for train_idx, test_idx in skf.split(X, y):

    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]

    X_train, X_test = standardize_train_test(X_train_raw, X_test_raw)

    model = LogisticRegression(lr=0.1, iterations=1500)
    model.fit(X_train, y[train_idx])

    # Predictions with different thresholds
    preds_05 = model.predict(X_test, threshold=0.5)
    preds_03 = model.predict(X_test, threshold=0.3)

    metrics_05 = calculate_metrics(y[test_idx], preds_05)
    metrics_03 = calculate_metrics(y[test_idx], preds_03)

    recall_05.append(metrics_05["Recall"])
    recall_03.append(metrics_03["Recall"])

    precision_05.append(metrics_05["Precision"])
    precision_03.append(metrics_03["Precision"])

# Take averages
recall_vals = [np.mean(recall_05), np.mean(recall_03)]
precision_vals = [np.mean(precision_05), np.mean(precision_03)]

labels = ["Threshold 0.5", "Threshold 0.3"]
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width/2, recall_vals, width, label="Recall")
plt.bar(x + width/2, precision_vals, width, label="Precision")

plt.xticks(x, labels)
plt.ylabel("Score")
plt.title("Logistic Regression: Effect of Threshold on Recall and Precision")
plt.legend()
plt.grid(True)


# Add labels for recall bars
for index in range(len(recall_vals)):
    recall_value = recall_vals[index]
    x_position = index - width / 2
    y_position = recall_value + 0.01

    plt.text(x_position, y_position, f"{recall_value:.3f}", ha='center')

# Add labels for precision bars
for index in range(len(precision_vals)):
    precision_value = precision_vals[index]
    x_position = index + width / 2
    y_position = precision_value + 0.01

    plt.text(x_position, y_position, f"{precision_value:.3f}", ha='center')

plt.show()


