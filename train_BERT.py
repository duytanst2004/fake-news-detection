import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW  # Updated import
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load and preprocess data
def load_data(fake_path, true_path):
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Add labels: 0 for fake, 1 for true
    df_fake['Label'] = 0
    df_true['Label'] = 1

    # Combine and shuffle
    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1).reset_index(drop=True)

    # Handle missing values and combine title and text
    df = df.fillna('')
    df['text'] = df['title'] + ' ' + df['text']

    # Keep only text and label
    return df[['text', 'Label']]

# Prepare data for BERT
def prepare_data(texts, labels, tokenizer, max_length=100, batch_size=32):
    encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels.values))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# Train the model
def train_model(model, train_loader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        for batch in tqdm(train_loader):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


# Evaluate the model
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return true_labels, predictions


def main():
    # Paths to dataset (update based on your Kaggle environment)
    fake_path = '/kaggle/input/fake-news-datasets/data/Fake.csv'
    true_path = '/kaggle/input/fake-news-datasets/data/True.csv'

    # Load and preprocess data
    df = load_data(fake_path, true_path)

    # Split data
    X = df['text']
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Prepare data loaders
    train_loader = prepare_data(X_train, y_train, tokenizer)
    test_loader = prepare_data(X_test, y_test, tokenizer)

    # Initialize model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Set optimizer
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train model
    train_model(model, train_loader, optimizer, device)

    # Evaluate model
    true_labels, predictions = evaluate_model(model, test_loader, device)
    print(classification_report(true_labels, predictions, target_names=['Fake', 'Real']))

    # Save model
    model.save_pretrained('./BERT_model')
    tokenizer.save_pretrained('./BERT_model')

if __name__ == '__main__':
    main()