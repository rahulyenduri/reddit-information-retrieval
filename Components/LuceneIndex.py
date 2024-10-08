import logging, sys

logging.disable(sys.maxsize)

import argparse
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
import json
import time
import matplotlib.pyplot as plt


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

            for reply in comment.get("replies", []):
                doc.add(Field("replies_to_comment", reply["body"], idType))

        writer.addDocument(doc)

    writer.close()

    end_time = time.time()
    return end_time - start_time  # Return the time taken to index


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
            "comment_body": doc.get("comment_body"),
            "replies_to_comment": doc.get("replies_to_comment"),
            # You can add more fields if necessary
        })

    return topkdocs


def analyze_index_timing(reddit_data_list):
    data_sizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]  # Example subset sizes
    times = []

    for size in data_sizes:
        subset = reddit_data_list[:size]
        time_taken = create_index('reddit_lucene_index_' + str(size), subset)
        times.append(time_taken)
        print(f"Indexing {size} documents took {time_taken:.2f} seconds")

    plt.plot(data_sizes, times, marker='o')
    plt.title('Lucene Indexing Performance')
    plt.xlabel('Number of Documents')
    plt.ylabel('Time Taken (seconds)')
    plt.grid(True)
    plt.show()


parser = argparse.ArgumentParser(description='Run Lucene Indexing and Retrieval')
parser.add_argument('--inputJsonFile', required=True, help='Path to the input JSON file')
parser.add_argument('--queryKeyword', required=True, help='Query keyword for retrieval')
parser.add_argument('--numberOfReturnedDocs', type=int, required=True, help='Number of documents to return')
args = parser.parse_args()


lucene.initVM(vmargs=['-Djava.awt.headless=true'])
with open(args.inputJsonFile, 'r', encoding='utf-8') as file:
    reddit_data = json.load(file)

create_index('reddit_lucene_index/', reddit_data)
results = retrieve('reddit_lucene_index/', args.queryKeyword, args.numberOfReturned)
print(results)


# analyze_index_timing(reddit_data)
