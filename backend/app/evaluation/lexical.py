"""Deterministic BM25 baseline with no model or network dependency."""
from collections import Counter, defaultdict
import re

import numpy as np


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


class BM25:
    def __init__(self, texts: list[str]):
        self.size = len(texts)
        self.lengths = np.array([len(tokens(text)) for text in texts], dtype=float)
        self.average_length = float(self.lengths.mean()) if texts else 1.0
        self.postings = defaultdict(list)
        for index, text in enumerate(texts):
            for term, count in Counter(tokens(text)).items():
                self.postings[term].append((index, count))

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(self.size, dtype=float)
        for term in set(tokens(query)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            idf = np.log(1 + (self.size - len(posting) + 0.5) / (len(posting) + 0.5))
            for index, count in posting:
                norm = 1.2 * (0.25 + 0.75 * self.lengths[index] / max(self.average_length, 1))
                scores[index] += idf * count * 2.2 / (count + norm)
        return scores

    def rank(self, query: str) -> list[int]:
        return np.argsort(-self.scores(query), kind="stable").tolist()
