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

results = retrieve('reddit_lucene_index/', args.queryKeyword, args.numberOfReturned)
print(results)