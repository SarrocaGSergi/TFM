import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = ["I'm so happy I'm doing this",
              "This is nuts"]

tokens_tensors = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')
print("Tokens:", tokens_tensors['input_ids'])

outputs = model(**tokens_tensors)
print("Logits:", outputs.logits)

probs = F.softmax(outputs.logits, dim=1)
print("Probs:", probs)





