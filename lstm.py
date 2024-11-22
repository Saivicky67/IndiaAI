import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Load the train and test data with the specified encoding
train_data = pd.read_csv(r'C:\Users\91861\Downloads\train1.csv', encoding='latin-1')
test_data = pd.read_csv(r'C:\Users\91861\Downloads\test1.csv', encoding='latin-1')

# Fill missing values with empty strings
train_data['crimeaditionalinfo'] = train_data['crimeaditionalinfo'].fillna("")
test_data['crimeaditionalinfo'] = test_data['crimeaditionalinfo'].fillna("")

# Combine the datasets for consistent preprocessing
combined_data = pd.concat([train_data, test_data])

# Filter classes with fewer than two members
class_counts = combined_data['category'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
combined_data = combined_data[combined_data['category'].isin(valid_classes)]

# Encode the labels
label_encoder = LabelEncoder()
combined_data['category'] = label_encoder.fit_transform(combined_data['category'])

# Split the data back into train and test sets
train_data, test_data = train_test_split(combined_data, test_size=0.2, stratify=combined_data['category'], random_state=42)

# Tokenize the text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_data['crimeaditionalinfo'].tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_data['crimeaditionalinfo'].tolist(), truncation=True, padding=True, max_length=128)

class CustomDataset(Dataset):
  def __init__(self, encodings, labels):
    self.encodings = encodings
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# Create datasets
train_dataset = CustomDataset(train_encodings, train_data['category'].tolist())
test_dataset = CustomDataset(test_encodings, test_data['category'].tolist())

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

class LSTMClassifier(nn.Module):
  def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
    super(LSTMClassifier, self).__init__()
    self.embedding = nn.Embedding(input_dim, hidden_dim)
    self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
    self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    embedded = self.dropout(self.embedding(x))
    lstm_output, (hidden, cell) = self.lstm(embedded)
    if self.lstm.bidirectional:
      hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
    else:
      hidden = self.dropout(hidden[-1,:,:])
    output = self.fc(hidden)
    return output

input_dim = tokenizer.vocab_size
hidden_dim = 256
output_dim = len(label_encoder.classes_)
n_layers = 2
bidirectional = True
dropout = 0.5

model = LSTMClassifier(input_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)

def train(model, loader, optimizer, criterion):
  model.train()
  epoch_loss = 0
  for batch in loader:
    optimizer.zero_grad()
    predictions = model(batch['input_ids']).squeeze(1)
    loss = criterion(predictions, batch['labels'])
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  return epoch_loss / len(loader)

def evaluate(model, loader, criterion):
  model.eval()
  epoch_loss = 0
  preds = []
  with torch.no_grad():
    for batch in loader:
      predictions = model(batch['input_ids']).squeeze(1)
      loss = criterion(predictions, batch['labels'])
      epoch_loss += loss.item()
      preds.extend(torch.argmax(predictions, 1).cpu().numpy())
  return epoch_loss / len(loader), preds

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

n_epochs = 5

for epoch in range(n_epochs):
  train_loss = train(model, train_loader, optimizer, criterion)
  val_loss, preds = evaluate(model, test_loader, criterion)
  acc = accuracy_score(test_data['category'].tolist(), preds)
  print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {acc:.3f}')

# Final Evaluation
val_loss, preds = evaluate(model, test_loader, criterion)
acc = accuracy_score(test_data['category'].tolist(), preds)
print(f'Final Accuracy: {acc:.3f}')

# Classification Report
report = classification_report(test_data['category'].tolist(), preds, target_names=label_encoder.classes_)
print(f'Classification Report:\n{report}')

