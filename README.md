# reddit-information-retrieval

This project explores an innovative approach to indexing and querying Reddit data using both traditional Lucene indexing and advanced BERT embeddings, integrated with Faiss for efficient similarity search. The project demonstrates the differences between sparse and dense indexing methods in handling information retrieval tasks.

## Introduction

The goal of this project is to create a system that indexes Reddit posts and performs information retrieval tasks using both Lucene and BERT embeddings. By employing BERT for dense vector representations and Lucene for sparse, keyword-based indexing, we aim to provide insights into the efficiency, accuracy, and performance of each method for retrieving relevant documents from a large dataset.

## Project Overview

The project consists of the following key components:

- BERT Embedding Module: Uses BERT embeddings to capture the semantic meaning of Reddit posts and store them in a Faiss index.
- Lucene Indexing Module: Uses Lucene for traditional text-based indexing with keyword search capabilities.
- Flask Web Application: Provides a user interface for querying both the BERT-Faiss index and the Lucene index, with the ability to compare search results.
- Reddit Data Crawler: Collects Reddit data, including post titles, selftext, comments, and more, for indexing.

## Technologies Used

- BERT (via Hugging Face Transformers) for dense text embeddings
- Faiss for efficient similarity search over embeddings
- Lucene (PyLucene) for traditional text-based indexing and retrieval
- Flask for the web interface
- PyTorch for generating BERT embeddings
- PRAW for Reddit data crawling

## System Architecture

The system has two major indexing methods:

1. BERT + Faiss: Uses BERT embeddings for dense vector representation and Faiss for fast similarity-based search.
2. Lucene: Uses keyword-based sparse vector indexing for fast keyword search and retrieval.

### Components:

- Flask Web Application: User interface for submitting search queries and selecting indexing options.
- BERT Embedding Module: Generates dense vector embeddings using BERT.
- Faiss Indexing Module: Stores and retrieves documents based on similarity search.
- Lucene Indexing Module: Indexes and retrieves documents using traditional Lucene indexing.
- Query Interface: Allows users to query both indices and retrieve relevant posts.

## Features

- Dual Indexing: Supports both sparse (Lucene) and dense (BERT + Faiss) indexing methods.
- Search Interface: A web-based search form allowing users to input queries and choose the indexing method.
- Efficient Search: Faiss enables high-speed similarity search over BERT embeddings.
- Query Results Ranking: Documents are ranked based on cosine similarity in Faiss and BM25 similarity in Lucene.
- Comparison: Allows users to compare search results across both indexing methods.

## Performance Comparison

### BERT + Faiss

- Advantages: Fast similarity search, great for semantic retrieval.
- Disadvantages: Slower indexing due to BERT’s computational overhead.

### Lucene

- Advantages: Excellent for keyword-based queries, efficient for large text-based datasets.
- Disadvantages: Less effective at capturing semantic relationships between terms.

## Limitations

1. Scalability: Indexing and retrieval may become resource-intensive with very large datasets, especially with BERT-based embeddings.
2. Resource Demand: The BERT model and Faiss indexing are computationally expensive, limiting performance on resource-constrained systems.
3. Indexing Overhead: Both Lucene and BERT indexing may introduce significant processing time for large datasets.
