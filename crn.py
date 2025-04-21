import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, auc

# Load dataset
csv_filename = "D:\spectrum_data_newrnn.csv"
df = pd.read_csv(csv_filename)

# Extract features and labels
X = df.iloc[:, 2:].values  # Features (excluding frequency and label columns)
y = df['Label'].values  # Labels

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define Leaky Integrated Bidirectional Echo State Network (LIBESN)
class LIBESN(nn.Module):
    def __init__(self, input_dim, reservoir_size, output_dim, leak_rate=0.3):
        super(LIBESN, self).__init__()
        self.reservoir_size = reservoir_size
        self.leak_rate = leak_rate
        
        # Randomly initialized reservoir weights
        self.W_in = nn.Linear(input_dim, reservoir_size, bias=False)
        self.W_res = nn.Linear(reservoir_size, reservoir_size, bias=False)
        self.W_out = nn.Linear(reservoir_size * 2, output_dim)  # Bidirectional
        
        # Initialize weights with small values
        nn.init.uniform_(self.W_res.weight, -0.5, 0.5)
        nn.init.uniform_(self.W_in.weight, -0.5, 0.5)
    
    def forward(self, x):
        batch_size = x.shape[0]
        res_states = torch.zeros(batch_size, self.reservoir_size)
        rev_states = torch.zeros(batch_size, self.reservoir_size)
        
        for t in range(x.shape[1]):
            res_states = (1 - self.leak_rate) * res_states + self.leak_rate * torch.tanh(self.W_in(x) + self.W_res(res_states))
            rev_states = (1 - self.leak_rate) * rev_states + self.leak_rate * torch.tanh(self.W_in(x) + self.W_res(rev_states))
        
        final_states = torch.cat((res_states, rev_states), dim=1)  # Combine forward and backward states
        output = self.W_out(final_states)
        return output

# Model initialization
input_dim = X_train.shape[1]
reservoir_size = 256600000000000002
output_dim = 2  # Binary classification
model = LIBESN(input_dim, reservoir_size, output_dim)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 150
loss_history = []  # Store loss values for plotting

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    
    loss_history.append(loss.item())  # Store loss for plotting
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Plot Training Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs + 1), loss_history, label='Training Loss', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid()
plt.show()

# Evaluate model
with torch.no_grad():
    y_pred = model(X_test).argmax(dim=1)
    
    # Accuracy
    accuracy = accuracy_score(y_test.numpy(), y_pred.numpy())
    
    # Precision, Recall, F1-Score
    precision, recall, f1, _ = precision_recall_fscore_support(y_test.numpy(), y_pred.numpy(), average='binary')

    # Print results
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1-Score: {f1:.4f}')
    
    # Confusion Matrix
    cm = confusion_matrix(y_test.numpy(), y_pred.numpy())
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Plot Precision-Recall Curve
with torch.no_grad():
    y_scores = torch.softmax(model(X_test), dim=1)[:, 1].numpy()  # Probabilities for positive class
    precision_vals, recall_vals, _ = precision_recall_curve(y_test.numpy(), y_scores)
    pr_auc = auc(recall_vals, precision_vals)

    plt.figure(figsize=(8, 5))
    plt.plot(recall_vals, precision_vals, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()
