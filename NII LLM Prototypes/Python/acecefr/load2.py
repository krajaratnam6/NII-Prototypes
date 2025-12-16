from transformers import BertModel, BertTokenizer
from torch import nn
import torch

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.pooler_output
            x = self.dropout(pooled_output)
            logits = self.fc(x)
            return logits

bert_model_name = 'bert-base-uncased'
num_classes = 6
max_length = 512

loaded_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = BERTClassifier(bert_model_name, num_classes).to(device)
loaded_model.load_state_dict(torch.load("bert_classifier_v3.pth", weights_only=True), strict=False)


def predict_cefr(text, model, tokenizer, device, max_length):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    return ["A1", "A2", "B1", "B2", "C1", "C2"][preds.item()]
