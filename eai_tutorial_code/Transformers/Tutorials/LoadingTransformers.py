import torch

from transformers import BertConfig, BertModel, AutoTokenizer

config = BertConfig()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertModel(config)

print(config)

############################################
#### ENCODING SENTENCES ####################
############################################

sequences = ["Hello!", "Cool", "Nice!"]

encoded_sequences = tokenizer(sequences,  padding=True, truncation=True, return_tensors='pt')
print("Encoded sequences: ", encoded_sequences)

otps = model(**encoded_sequences)

print("Outputs: ", otps)






