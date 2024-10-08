import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import json


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')  # you can change the model here
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')


def convert_to_embedding(query):
    tokens = {'input_ids': [], 'attention_mask': []}
    new_tokens = tokenizer.encode_plus(query, max_length=512,
                                       truncation=True, padding='max_length',
                                       return_tensors='pt')
    tokens['input_ids'].append(new_tokens['input_ids'][0])
    tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    with torch.no_grad():
        outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    return mean_pooled[0]  # assuming query is a single sentence


query = "python or java"
query_embedding = convert_to_embedding(query)

index_loaded = faiss.read_index("sample_code.index")
D, I = index_loaded.search(query_embedding[None, :], 4)

with open("../mergedAll.json", 'r', encoding='utf-8') as file:
    json_data = json.load(file)
print(I, json_data[np.argmax(I)]['selftext'])