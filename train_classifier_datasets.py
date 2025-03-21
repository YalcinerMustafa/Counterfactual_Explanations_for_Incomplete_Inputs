import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import torch.onnx
from enum import Enum

class Domain(Enum):
    CONCRETE = 1
    DIABETES = 2
    CANCER = 3
    WINE = 4
    MOONS = 5
    FETAL = 6
    POWER = 7

def train():
    # Load the dataset from a CSV file
    if domain == Domain.CONCRETE:
        data_folder = "datasets/concrete/split"
        MODEL_NAME = "concrete"
        file_path_train = f'{data_folder}/concrete_train.csv'
        file_path_test = f'{data_folder}/concrete_test.csv'
        file_path_val = f'{data_folder}/concrete_val.csv'
    elif domain == Domain.DIABETES:
        data_folder = "datasets/diabetes/split"
        MODEL_NAME = "diabetes"
        file_path_train = f'{data_folder}/diabetes_train.csv'
        file_path_test = f'{data_folder}/diabetes_test.csv'
        file_path_val = f'{data_folder}/diabetes_test.csv'
    elif domain == Domain.CANCER:
        data_folder = "datasets/breast_cancer/split"
        MODEL_NAME = "breast_cancer"
        file_path_train = f'{data_folder}/breast_cancer_train.csv'
        file_path_test = f'{data_folder}/breast_cancer_test.csv'
        file_path_val = f'{data_folder}/breast_cancer_val.csv'
    elif domain == Domain.WINE:
        data_folder = "datasets/wine/split"
        MODEL_NAME = "wine"
        file_path_train = f'{data_folder}/winequality_train.csv'
        file_path_test = f'{data_folder}/winequality_test.csv'
        file_path_val = f'{data_folder}/winequality_val.csv'
    elif domain == Domain.MOONS:
        data_folder = "datasets/moons/split"
        MODEL_NAME = "moons"
        file_path_train = f'{data_folder}/moons_train.csv'
        file_path_test = f'{data_folder}/moons_test.csv'
        file_path_val = f'{data_folder}/moons_val.csv'
    elif domain == Domain.FETAL:
        data_folder = "datasets/fetal/split"
        MODEL_NAME = "fetal"
        file_path_train = f'{data_folder}/train.csv'
        file_path_test = f'{data_folder}/test.csv'
        file_path_val = f'{data_folder}/val.csv'
    elif domain == Domain.POWER:
        data_folder = "datasets/power/split"
        MODEL_NAME = "power"
        file_path_train = f'{data_folder}/train.csv'
        file_path_test = f'{data_folder}/test.csv'
        file_path_val = f'{data_folder}/val.csv'


    df_train = pd.read_csv(file_path_train, delimiter=',')
    df_test = pd.read_csv(file_path_test, delimiter=',')
    df_val = pd.read_csv(file_path_val, delimiter=',')

    # Assume that all these datasets have a "target" column for the target.
    X_train = df_train.drop(columns="target")
    y_train = df_train["target"]
    X_test = df_test.drop(columns="target")
    y_test = df_test["target"]
    X_val = df_val.drop(columns="target")
    y_val = df_val["target"]



    class ClassifierMoons(nn.Module):
        def __init__(self):
            super(ClassifierMoons, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], X_train.shape[1])
            self.fc2 = nn.Linear(X_train.shape[1], 2)  # Number of classes in target

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x

    # Define the neural network model
    class Classifier(nn.Module):
        def __init__(self):
            super(Classifier, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], round(X_train.shape[1]/2))
            self.fc2 = nn.Linear(round(X_train.shape[1]/2), 2)  # Number of classes in target

        def forward(self, x):
            x = self.fc1(x)
            x = torch.relu(x)
            x = self.fc2(x)
            return x

    class LinearClassifier(nn.Module):
        def __init__(self):
            super(LinearClassifier, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 2)

        def forward(self, x):
            x = self.fc1(x)
            return x

    # Split data into training, validation, and test sets
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.to_numpy().astype(np.float32))
    y_train_tensor = torch.tensor(y_train.to_numpy().astype(np.int64))
    X_val_tensor = torch.tensor(X_val.to_numpy().astype(np.float32))
    y_val_tensor = torch.tensor(y_val.to_numpy().astype(np.int64))
    X_test_tensor = torch.tensor(X_test.to_numpy().astype(np.float32))
    y_test_tensor = torch.tensor(y_test.to_numpy().astype(np.int64))

    # Create PyTorch datasets and data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Initialize the model, loss function, and optimizer
    if TRAIN_DEEP:
        model = Classifier()
    else:
        model = LinearClassifier()

    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0

    # Train the model
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        # Calculate average training loss
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {avg_val_loss}')

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    # Load the best model
    model.load_state_dict(torch.load('best_model.pth'))

    # Evaluate the model on the test set
    model.eval()
    y_pred = []
    y_true = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=1)  # Handle undefined precision

    print(f'Classification Accuracy: {accuracy}')
    print('Classification Report:')
    print(report)

    if TRAIN_DEEP:
        # Export the trained model to ONNX
        onnx_file_path = f'{data_folder}/{MODEL_NAME}_medium.onnx'  # Output path for the ONNX model
        report_file_path = f'{data_folder}/confusion_matrix_medium.txt'
    else:
        onnx_file_path = f'{data_folder}/{MODEL_NAME}_tiny.onnx'  # Output path for the ONNX model
        report_file_path = f'{data_folder}/confusion_matrix_tiny.txt'
    with open(report_file_path, 'w') as file:
        file.write(f'Classification Accuracy: {accuracy}\n\n')
        file.write('Classification Report:\n')
        file.write(report)

    # Take the dummy input from training data. Otherwise, node types become "GEMM" or so.
    df_train_dummy = pd.read_csv(file_path_train, delimiter=',')
    X_dummy = df_train_dummy.iloc[:, :-1].values.astype(np.float32)  # Features
    y_dummy = df_train_dummy.iloc[:, -1].values.astype(np.int64)  # Target
    X_train_dummy, X_temp, y_train_dummy, y_temp_dummy = train_test_split(X_dummy, y_dummy, test_size=0.3,
                                                                          random_state=42)
    torch.onnx.export(model, torch.from_numpy(X_train_dummy[0]).float(), onnx_file_path, export_params=True,
                      verbose=True)

    print(f'Model has been exported as {onnx_file_path}')


if __name__ == '__main__':
    TRAIN_DEEP = True
    domain = Domain.POWER
    train()
