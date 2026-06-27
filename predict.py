import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle


def load_bert_model(model_path='BERT_model'):
    """Load pre-trained BERT model and tokenizer."""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device


def load_naive_bayes_model(model_path='naive_bayes_model.pkl'):
    """Load pre-trained Naive Bayes model and vectorizer."""
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data['model'], saved_data['vectorizer']


def predict_bert(text, model, tokenizer, device, max_length=100):
    """Predict using BERT model."""
    # Tokenize input text
    encodings = tokenizer([text], truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).cpu().numpy()[0]

    return 'Real' if prediction == 1 else 'Fake'


def predict_naive_bayes(text, model, vectorizer):
    """Predict using Naive Bayes model."""
    # Transform text using the saved vectorizer
    text_tfidf = vectorizer.transform([text])
    # Predict
    prediction = model.predict(text_tfidf)[0]
    return 'Real' if prediction == 1 else 'Fake'


def predict(text):
    """Predict using both BERT and Naive Bayes models."""
    # Load models
    bert_model, bert_tokenizer, device = load_bert_model()
    nb_model, nb_vectorizer = load_naive_bayes_model()

    # Get predictions
    bert_pred = predict_bert(text, bert_model, bert_tokenizer, device)
    nb_pred = predict_naive_bayes(text, nb_model, nb_vectorizer)

    return {
        'BERT': bert_pred,
        'Naive Bayes': nb_pred
    }

if __name__ == '__main__':
    # Example usage
    sample_text = "This is not a sample news article to test fake news detection."
    predictions = predict(sample_text)
    print("Predictions:")
    print(f"BERT: {predictions['BERT']}")
    print(f"Naive Bayes: {predictions['Naive Bayes']}")