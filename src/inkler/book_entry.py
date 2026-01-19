import glob
import logging
from typing import List, Dict
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import time

import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import torch
from torch.nn.functional import one_hot

from inkler.constants import N_GENRES


@dataclass
class BookEntry:
    title: str
    author: str
    text: np.ndarray
    score: float
    genre: np.ndarray
    hierarchy_ids: np.ndarray
    postit: np.ndarray | None = None


class BookBatch:
    def __init__(
            self,
            book_entry_l: List[BookEntry],
            scale: float = 0.01,
            dtype: torch.dtype = torch.float32
    ):
        self.titles = []
        self.authors = []
        self.texts = []
        self.scores = []
        self.genres = []
        self.postits = []
        self.hierarchy = []
        self.masks = []
        max_sentence = max(len(t.text) for t in book_entry_l)
        max_hierarchy = None
        for it, t in enumerate(book_entry_l):
            self.titles.append(t.title)
            self.authors.append(t.author)
            text = -np.ones((max_sentence, len(t.text[0])))
            text[:len(t.text)] = t.text
            self.texts.append(text)
            mask = np.zeros(max_sentence, dtype="bool")
            mask[len(t.text):] = True
            self.masks.append(mask)
            self.scores.append(t.score)
            self.genres.append(one_hot(torch.tensor(t.genre), num_classes=N_GENRES).sum(dim=0).clip(min=0))
            if max_hierarchy is None:
                max_hierarchy = np.zeros(len(t.hierarchy_ids[0]))
            hierarchy  = -np.ones((max_sentence, len(max_hierarchy)))
            hierarchy[:len(t.hierarchy_ids)] = np.array(t.hierarchy_ids) + max_hierarchy
            self.hierarchy.append(np.vstack([it * np.ones(max_sentence), hierarchy.T]).T)
            max_hierarchy = max_hierarchy + np.max(t.hierarchy_ids, axis=0)
            self.postits.append(t.postit if t.postit is not None else -np.inf * np.ones(len(t.text[0])))

        self.genres = torch.tensor(np.vstack(self.genres)).to(dtype)
        self.scores = torch.tensor(np.array(self.scores)).to(dtype) * scale
        self.texts = torch.tensor(np.stack(self.texts)).to(dtype)
        self.hierarchy = torch.tensor(np.stack(self.hierarchy)).to(dtype)
        self.postits = torch.tensor(np.vstack(self.postits)).unsqueeze(1).to(dtype)
        self.masks = torch.tensor(np.stack(self.masks)).to(dtype)

    def get_dict(self):
        return {
            "titles": self.titles,
            "authors": self.authors,
            "texts": self.texts,
            "scores": self.scores,
            "genres": self.genres,
            "postits": self.postits,
            "hierarchy": self.hierarchy,
            "masks": self.masks,
        }


def _load(path: str | Path):
    with open(path, "r") as f:
        return json.load(f)


class BookDataset:
    def __init__(
            self,
            train_path: List[str | Path] | str | Path,
            in_dim: int = 748,
            seq_len: int = 512,
            batch_size: int = 32,
            preload_size: int = 64,
            in_order: bool = False,
            n_cpu: int | None = None,
            device: int | str | torch.device = "cpu"
    ):
        if not isinstance(train_path, list):
            train_path = glob.glob(f"{train_path}/*.json")
        self.book_entry_path_l = train_path
        if batch_size is None:
            batch_size = 1
        elif batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        if preload_size is None:
            preload_size = 1
        elif preload_size < 1:
            raise ValueError("preload_size must be positive")
        elif preload_size < self.batch_size:
            preload_size = self.batch_size
        self.preload_size = preload_size
        if n_cpu is None or n_cpu == 0:
            n_cpu = 1
        elif n_cpu < 0:
            n_cpu = mp.cpu_count()
        self.n_cpu = min(n_cpu, self.preload_size)

        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        self.seq_len = seq_len
        self.doc_token = torch.finfo(torch.float16).min * torch.ones(in_dim)

        self.in_order = in_order
        if self.in_order:
            self.idx = np.arange(len(self.book_entry_path_l))
        else:
            self.idx = np.random.permutation(len(self.book_entry_path_l))
        self.pending_book_entries = []
        self.pool = ThreadPool(processes=self.n_cpu)
        self.epoch = 0
        self.current_idx = 0
        self.device = device
        self.preload_book_entries()

    def _reset(self):
        self.current_idx = 0
        self.epoch += 1
        if not self.in_order:
            np.random.shuffle(self.idx)

    def preload_book_entries(self):
        while len(self.pending_book_entries) < self.preload_size:
            if self.current_idx >= len(self.book_entry_path_l):
                self._reset()
            self.pending_book_entries.append(
                self.pool.apply_async(_load, (self.book_entry_path_l[self.idx[self.current_idx]],))
            )
            self.current_idx += 1

    def __len__(self):
        return len(self.book_entry_path_l)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type in [KeyboardInterrupt, SystemExit, SystemExit]:
            self.pool.terminate()

    def __iter__(self):
        return self

    def __next__(self):
        results = []
        if self.in_order:
            for i in range(self.batch_size):
                results.append(BookEntry(**self.pending_book_entries[i].get()))
            self.pending_book_entries = self.pending_book_entries[self.batch_size:]
        else:
            while len(results) < self.batch_size:
                for i_be in range(len(self.pending_book_entries) - 1, -1, -1):
                    if self.pending_book_entries[i_be].ready():
                        results.append(BookEntry(**self.pending_book_entries[i_be].get()))
                        del self.pending_book_entries[i_be]
                        if len(results) >= self.batch_size:
                            break
                time.sleep(0.01)

        self.preload_book_entries()
        return BookBatch(results).get_dict()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        return BookBatch([_load(self.book_entry_path_l[self.idx[i]]) for i in idx]).get_dict()






