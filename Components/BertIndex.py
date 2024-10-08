import time

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor


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


def split_into_segments(text, max_words=400):
    words = text.split()
    segments = [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
    return segments


def encode_text_segments(segments):
    # Consider splitting the segments into manageable batches if they are too many
    # For example:
    batch_size = 64  # Adjust based on your system's capabilities
    all_mean_pooled = []

    for i in range(0, len(segments), batch_size):
        batch_segments = segments[i:i + batch_size]
        tokens = tokenizer(batch_segments, padding=True, truncation=True, max_length=512, return_tensors='pt')

        with torch.no_grad():
            outputs = model(**tokens)

        embeddings = outputs.last_hidden_state
        attention_mask = tokens['attention_mask'].unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * attention_mask
        summed = torch.sum(masked_embeddings, 1)
        summed_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
        mean_pooled = summed / summed_mask
        all_mean_pooled.append(mean_pooled)

    # Return the average embedding of all mean-pooled batches
    return torch.cat(all_mean_pooled, dim=0).mean(dim=0)


def process_item(item):
    all_segments = []

    # Collect segments from 'selftext' and 'title'
    for field in ['selftext', 'title']:
        if field in item and item[field]:
            all_segments.extend(split_into_segments(item[field]))

    # Collect segments from comments
    for comment in item.get('comments', []):
        comment_body = comment.get('body', '')
        if comment_body:
            all_segments.extend(split_into_segments(comment_body))

    # If there are segments to process
    if all_segments:
        # Process each segment individually and store their embeddings
        segment_embeddings = [encode_text_segments([segment]) for segment in all_segments]

        # Then average those embeddings to get a single embedding for the item
        if segment_embeddings:
            item_embedding = torch.stack(segment_embeddings).mean(dim=0)
            return item_embedding.cpu()  # Move to CPU to avoid potential CUDA out-of-memory errors
    return None


def convert_json_embedding_multithreaded(json_data, num_threads=None):
    all_embeddings = []

    # ThreadPoolExecutor context manager
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Map process_item function to all items in json_data
        results = executor.map(process_item, json_data)

        # Iterate through results and append non-None embeddings
        for embedding in results:
            if embedding is not None:
                all_embeddings.append(embedding)

    # Stack all embeddings into a tensor if there are any, else return an empty tensor
    return torch.stack(all_embeddings)


def convert_json_embedding(json_data):
    all_embeddings = []

    for item in json_data:
        item_embeddings = []

        # Encode specified fields
        for field in ['selftext', 'title']:
            if field in item and item[field]:
                segments = split_into_segments(item[field])

                field_embedding = encode_text_segments(segments)
                item_embeddings.append(field_embedding)

        # Encode comments
        for comment in item.get('comments', []):
            comment_body = comment.get('body', '')
            if comment_body:
                comment_segments = split_into_segments(comment_body)

                comment_embedding = encode_text_segments(comment_segments)
                item_embeddings.append(comment_embedding)

        # Aggregate embeddings for the item
        if item_embeddings:
            item_embedding = torch.stack(item_embeddings).mean(dim=0)
            all_embeddings.append(item_embedding)

    # Stack all item embeddings to create a tensor
    return torch.stack(all_embeddings)


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')  # you can change the model here
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')


print("Loading Data...")
with open("../mergedAll.json", 'r', encoding='utf-8') as file:
    json_data = json.load(file)

data = json_data
print("Loading Data Finished")

# print("Encoding...")
# start_time = time.time()
# num_threads = 4  # Adjust based on your system's capabilities and the nature of your tasks
# mean_pooled = convert_json_embedding_multithreaded(data, num_threads)
# # mean_pooled = convert_json_embedding(json_data[:100])
# print(mean_pooled)
# print(f"Encoding time is: {time.time() - start_time}")
# print("Encoding Finished")

index_name = f"data_{len(data)}.index"
# print("Indexing...")
# start_time = time.time()
# index = faiss.IndexFlatIP(384)
# index.add(mean_pooled)
# faiss.write_index(index, index_name)
# duration = time.time() - start_time
# print("Indexing Finished")
# with open('timings.log', 'a') as log_file:
#     log_file.write(f'Indexing with faiss and # data {len(data)} $: {duration}\n')

query = "python web development"
query_embedding = convert_to_embedding(query)

print("Retreving")
index_loaded = faiss.read_index(index_name)

D, I = index_loaded.search(query_embedding[None, :], 4)

# Retrieve the results from json_data using the indices from FAISS
print(I)
print(D)
for idx_array in I:
    for idx in idx_array:
        result = json_data[idx]
        print(result)