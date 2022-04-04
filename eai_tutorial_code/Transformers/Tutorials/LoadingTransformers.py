import torch
import numpy as np
from transformers import BertConfig, BertModel, AutoTokenizer, AutoFeatureExtractor, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")
model = AutoModel.from_pretrained("sentence-transformers/distilbert-base-nli-mean-tokens")

sequences = ["POCKET SQUARES & TIE BARS", "BOOTS", "BOOTS"]
############################################
#### ENCODING SENTENCES ####################
############################################
encoded_tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors='pt')

# Compute Token embeddings

with torch.no_grad():
    model_output = model(**encoded_tokens)

# Pooling the result.

sequence_embeddings = mean_pooling(model_output, encoded_tokens['attention_mask'])

print("Sequence Embeddings:")
print(sequence_embeddings)






