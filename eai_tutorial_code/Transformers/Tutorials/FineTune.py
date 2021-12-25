import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

torch.cuda.empty_cache()

# Initialize the model
checkpoint = "bert-base-cased"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Raw Inputs
raw_inputs = ["I'm so happy I'm doing this",
              "This is bullshit"]

# Tokenize Data
token_tensors = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors='pt')

# Train the model
token_tensors["labels"] = torch.tensor([1, 1])
optimizer = AdamW(model.parameters())
loss = model(**token_tensors).loss
loss.backward()
optimizer.step()



