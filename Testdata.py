#Author: Patrick Tan
#contact: patrick.patricktan@gmail.com
#this training will train accuracy of FT001, evaluation as 0.
#这是无需运行的代码。测试记录而已

from flask import Flask, jsonify, render_template, send_from_directory
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset

app = Flask(__name__)

# Load data from CSV file
data = pd.read_csv('survey.csv')

# Preprocessing
X = data[['Texture', 'Crispiness', 'Taste', 'Visual Presentation', 'Overall Satisfaction']]
y = data['French Toast']

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert labels to tensor
y = y.map({'FT001': 0}).to_numpy()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define datasets
def create_datasets(X_train, X_test, y_train, y_test):
    train_encodings = tokenizer(X_train['Visual Presentation'].astype(str).tolist(), padding=True, truncation=True,
                                return_tensors='pt')
    test_encodings = tokenizer(X_test['Visual Presentation'].astype(str).tolist(), padding=True, truncation=True,
                               return_tensors='pt')

    train_labels = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    test_labels = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)

    return train_dataset, test_dataset


# Define the model architecture
class SurveyClassifier(nn.Module):
    def __init__(self):
        super(SurveyClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 1)  # Output dimension is 1 for binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    losses = []
    accuracies = []
    model.train()
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        predictions = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (predictions == labels).float().mean().item()
        accuracies.append(accuracy)
        losses.append(loss.item())

    return losses, accuracies


# Evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            logits = model(input_ids, attention_mask)
            predictions = (torch.sigmoid(logits) > 0.5).float()
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    return correct / total


# Flask routes
@app.route('/')
def index():
    return "Welcome to the Test Data Centre!"


@app.route('/train.html')
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SurveyClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    train_dataset, test_dataset = create_datasets(X_train, X_test, y_train, y_test)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    epochs = 5
    all_losses = []
    all_accuracies = []

    for epoch in range(epochs):
        losses, accuracies = train_model(model, train_loader, criterion, optimizer, device)
        epoch_loss = sum(losses) / len(losses)
        epoch_accuracy = sum(accuracies) / len(accuracies)
        all_losses.extend(losses)
        all_accuracies.extend(accuracies)

    accuracy = evaluate_model(model, test_loader, device)

    # Plotting loss and accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(all_losses)
    plt.title('Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(all_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')

    # Save the plot to a file
    plt.savefig('static/training_plot.png')

    # Prepare the HTML to display the records processed during training
    records_html = ""
    for i in range(len(all_losses)):
        records_html += f"<p>Iteration {i + 1}: Loss = {all_losses[i]}, Accuracy = {all_accuracies[i]}</p>"

    return render_template('train.html', accuracy=accuracy, records=records_html)


# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static'), path)


if __name__ == '__main__':
    app.run(debug=True)
