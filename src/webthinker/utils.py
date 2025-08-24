"""Utility functions."""

import asyncio
import json
import logging
import string
from typing import Dict, List, Set

import nltk
import numpy as np
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from mrkdwn_analysis import MarkdownAnalyzer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi


def get_logger(
    name: str,
    log_file: str = None,
) -> logging.Logger:
    """Get logger."""
    logger = logging.getLogger(name)
    if (log_file is not None) and (not logger.handlers):
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        logger.addHandler(file_handler)
    return logger


def extract_outline(content: str) -> str:
    """Extract outline of the article"""
    analyzer = MarkdownAnalyzer.from_string(content)
    results = analyzer.identify_headers()
    outline = ""
    for header in results["Header"]:
        outline += f"{'#' * header['level']} {header['text']}\n"
    return outline


def search_google_serper(
    query: str,
    max_results: int
) -> List[Dict[str, str]]:
    """Search query from google."""
    # Search results
    search = GoogleSerperAPIWrapper(
        type="search",  # organic
        k=max_results,
    )
    results = search.results(query)

    # Convert results
    if "organic" in results:
        search_results = [
            {
                "id": result["position"],
                "title": result["title"],
                "url": result["link"],
                "snippet": result["snippet"],
                "content": ""
            }
            for result in results["organic"]
        ]
    else:
        search_results = []
    return search_results


def search_tavily(
    query: str,
    max_results: int
) -> List[Dict[str, str]]:
    """Search query from tavily."""
    search = TavilySearchAPIWrapper()
    results = search.results(
        query=query,
        max_results=max_results,
    )
    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return [
        {
            "id": i,
            "title": result["title"],
            "url": result["url"],
            "snippet": result["content"],
            "content": ""
        }
        for i, result in enumerate(results)
    ]


def fetch_content(url: str) -> str:
    """Fetch webpage content."""
    async def fetch_content_async(url: str) -> str:
        """Async function to fetch webpage content."""
        browser_config = BrowserConfig(
            verbose=True,
            text_mode=True,
        )
        run_config = CrawlerRunConfig(
            verbose=True,
            scan_full_page=True,
            # exclude_external_links=True,
        )
        async with AsyncWebCrawler(
            config=browser_config,
        ) as crawler:
            result = await crawler.arun(
                url,
                config=run_config,
            )
        if result.success and result.markdown.strip():
            md_content = result.markdown.strip()
        else:
            md_content = "Can not fetch the page content."
        return md_content
    return asyncio.run(fetch_content_async(url))


def bag_of_words(sent: str) -> Set[str]:
    """Convert a sentence into a bag of words."""
    # Remove punctuation in sentence
    sent = sent.translate(str.maketrans("", "", string.punctuation))
    return set(sent.lower().split())


def f1_score(true_set: Set[str], pred_set: Set[str]) -> float:
    """Compute F1 score of two sets."""
    intersection = true_set.intersection(pred_set)
    if intersection:
        p = len(intersection) / len(pred_set)
        r = len(intersection) / len(true_set)
        return 2 * p * r / (p + r)
    return 0.


def extract_context_by_snippet(
    raw_content: str,
    snippet: str,
    context_chars: int = 4000,
) -> str:
    """Extract the context from document via snippet."""
    # Split sentences
    sents = nltk.sent_tokenize(raw_content)
    snippet_words = bag_of_words(snippet)
    # Main idea is to find the most relevant sentence and expand the context
    best_f1 = 0.2
    best_sent_id = -1
    for i, sent in enumerate(sents):
        sent = sent.translate(str.maketrans("", "", string.punctuation))
        sent_words = bag_of_words(sent)
        f1 = f1_score(snippet_words, sent_words)
        if f1 > best_f1:
            best_f1, best_sent_id = f1, i

    if best_sent_id >= 0:
        key_sent = sents[best_sent_id]
        sent_start = raw_content.find(key_sent)
        start_idx = max(0, sent_start - context_chars)
        return raw_content[start_idx:start_idx + len(key_sent) + context_chars]

    return raw_content[:2 * context_chars]


def format_search_results(
    search_results: List[Dict[str, str]],
    with_content: bool = True,
) -> str:
    """Format search results into readable string."""
    formatted_results = ""
    for i, result in enumerate(search_results):
        formatted_results += f"***Web Page {i+1}:***\n"
        page_info = {
            "title": result["title"],
            "url": result["url"],
            "snippet": result["snippet"],
        }
        if with_content:
            page_info["content"] = result["content"]
        formatted_results += json.dumps(page_info, indent=2, ensure_ascii=False)
        formatted_results += "\n"
    return formatted_results


class BM25Retriever:
    """Modify BM25Retriever to support add_documents."""

    def __init__(self):
        self.docs = []
        self.tokenized_docs = []
        self.bm25 = None

    def add_documents(self, docs: List[str]):
        """Add new documents to the retriever."""
        # rank_bm25 does not support incremental updates, so we need to build all indices
        self.docs.extend(docs)
        self.tokenized_docs.extend([word_tokenize(doc.lower()) for doc in docs])
        if self.tokenized_docs:
            self.bm25 = BM25Okapi(self.tokenized_docs)
        else:
            self.bm25 = None

    def invoke(self, query: str, k: int = 3) -> List[str]:
        """Retrieve top k documents for a given query."""
        if self.bm25 is None:
            return []

        tokenized_query = word_tokenize(query.lower())
        doc_scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(doc_scores)[-k:][::-1]
        return [self.docs[i] for i in top_indices]
