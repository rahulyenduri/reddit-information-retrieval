from flask import Flask, request, render_template

import logging, sys
import time

logging.disable(sys.maxsize)

from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import json
from flask import jsonify

import lucene
import os
from org.apache.lucene.store import MMapDirectory, SimpleFSDirectory, NIOFSDirectory
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, IntPoint, StoredField
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions, DirectoryReader
from org.apache.lucene.search import IndexSearcher, BoostQuery, Query
from org.apache.lucene.search import BooleanQuery, BooleanClause
from org.apache.lucene.search.similarities import BM25Similarity


tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5')
model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5')
app = Flask(__name__)


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

    return mean_pooled[0]


def BertInference(query, number_results):
    print("Loading Data...")
    with open("../mergedAll.json", 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    query_embedding = convert_to_embedding(query)

    print("Retreving")
    index_loaded = faiss.read_index("data_34641.index")

    D, I = index_loaded.search(query_embedding[None, :], number_results)

    results = []  # Initialize an empty list to store your results

    # Retrieve the results from json_data using the indices from FAISS
    # print(I)
    for idx_array in I:
        for idx in idx_array:
            result = json_data[idx]
            # print(result)
            results.append(result)  # Append each result to the results list

    return results


def retrieve(storedir, query, numberOfReturned):
    searchDir = NIOFSDirectory(Paths.get(storedir))
    searcher = IndexSearcher(DirectoryReader.open(searchDir))
    analyzer = StandardAnalyzer()

    # Construct a query for each field you want to search in
    titleParser = QueryParser("title", analyzer)
    titleQuery = titleParser.parse(query)
    selftextParser = QueryParser("selftext", analyzer)
    selftextQuery = selftextParser.parse(query)
    commentBodyParser = QueryParser("comment_body", analyzer)
    commentBodyQuery = commentBodyParser.parse(query)

    # Combine the queries into a single BooleanQuery
    combinedQuery = BooleanQuery.Builder()
    combinedQuery.add(titleQuery, BooleanClause.Occur.SHOULD)  # SHOULD means this clause is optional, but contributes to the score
    combinedQuery.add(selftextQuery, BooleanClause.Occur.SHOULD)  # Same here
    combinedQuery.add(commentBodyQuery, BooleanClause.Occur.SHOULD)  # And here

    topDocs = searcher.search(combinedQuery.build(), numberOfReturned).scoreDocs  # Adjust the number of top docs as needed
    topkdocs = []
    for hit in topDocs:
        doc_id = hit.doc
        doc_score = hit.score
        doc = searcher.doc(doc_id)
        topkdocs.append({
            "score": doc_score,
            "title": doc.get("title"),
            "selftext": doc.get("selftext"),
            "id": doc.get("id"),
            "url": doc.get("url"),
            "permalink": doc.get("permalink"),
            "comments": [doc.get("comment_body")],
            # "replies_to_comment": doc.get("replies_to_comment"),
            # You can add more fields if necessary
        })

    print(topkdocs)
    return topkdocs

def create_index(dir, reddit_data):
    start_time = time.time()
    if not os.path.exists(dir):
        os.mkdir(dir)
    store = SimpleFSDirectory(Paths.get(dir))
    analyzer = StandardAnalyzer()
    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
    writer = IndexWriter(store, config)

    idType = FieldType()
    idType.setStored(True)
    idType.setTokenized(False)

    textType = FieldType()
    textType.setStored(True)
    textType.setTokenized(True)
    textType.setIndexOptions(IndexOptions.DOCS)

    bodyType = FieldType()
    bodyType.setStored(True)
    bodyType.setTokenized(True)
    bodyType.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)

    numType = FieldType()
    numType.setStored(True)
    numType.setTokenized(False)

    for post in reddit_data:
        doc = Document()
        doc.add(Field("title", post["title"], textType))
        doc.add(Field("selftext", post["selftext"], bodyType))
        doc.add(Field("selftext", post["selftext"], bodyType))
        doc.add(Field("id", post["id"], idType))
        doc.add(Field("score", post["score"], numType))
        doc.add(Field("url", post["url"], idType))
        doc.add(Field("permalink", post["permalink"], idType))

        # Indexing comments
        for comment in post.get("comments", []):
            doc.add(Field("comment_body", comment["body"], bodyType))

            # for reply in comment.get("replies", []):
            #     doc.add(Field("replies_to_comment", reply["body"], idType))

        writer.addDocument(doc)

    writer.close()

    end_time = time.time()
    return end_time - start_time


def LuceneInference(query, number_results):

    try:
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    except:
        print("Can't run twice!")

    print("Loading Data...")
    with open("../mergedAll.json", 'r', encoding='utf-8') as file:
        reddit_data = json.load(file)

    create_index('reddit_lucene_index/', reddit_data)
    print("Retreving")
    results = retrieve('reddit_lucene_index/', query, number_results)

    return results


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    indexing_option = request.form['indexingOption']
    number_results = request.form['numberOfResults']

    if indexing_option.lower() == "dense":
        results = BertInference(str(query), int(number_results))
        print(results)
        return jsonify(results=results)  # Return results as JSON
    elif indexing_option.lower() == "sparse":
        results = LuceneInference(str(query), int(number_results))
        print(results)
        return jsonify(results=results)  # Return results as JSON

    # Handle other cases or error
    return jsonify(error="Unsupported indexing option")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)