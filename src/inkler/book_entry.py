import glob
from typing import List, Any
from pathlib import Path
from dataclasses import dataclass
import json
import numpy as np
import time

import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import torch
from torch.nn.functional import one_hot

from inkler.constants import N_GENRES, BUCKETS


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
            seq_len: int | None = None,
            scale: float = 1.,
            dtype: torch.dtype = torch.float32,
            thresh: float | None = None,
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
        dist_buckets = np.array(BUCKETS, dtype=np.float32) - max_sentence
        dist_buckets[dist_buckets < 0] = np.inf
        # get first bucket that is larger than the max text length
        max_sentence = BUCKETS[np.abs(dist_buckets).argmin()]
        max_hierarchy = None
        for it, t in enumerate(book_entry_l):
            hid_array = np.array(t.hierarchy_ids, dtype=np.int32)
            for i_hid, hid in enumerate(hid_array.T):
                hid_array[:, i_hid] = np.unique(hid, return_inverse=True)[1]
                if np.max(hid) + 1 > len(np.unique(hid)):
                    raise ValueError(f"There are hierarchy gaps for {t.title} by {t.author}")

            self.titles.append(t.title)
            self.authors.append(t.author)
            text = np.full((max_sentence, len(t.text[0])), -1, dtype=np.float32)
            text[:len(t.text)] = t.text
            self.texts.append(text)

            mask = np.zeros(max_sentence, dtype="bool")
            mask[len(t.text):] = True
            self.masks.append(mask)

            self.scores.append(t.score)
            self.genres.append(
                one_hot(torch.tensor(t.genre, dtype=torch.long), num_classes=N_GENRES).sum(dim=0).clip(min=0, max=1)
                if t.genre is not None and len(t.genre) > 0 else
                torch.zeros(N_GENRES, dtype=torch.long)
            )
            if max_hierarchy is None:
                max_hierarchy = np.zeros(hid_array.shape[1], dtype=np.float32)
            hierarchy = np.full((max_sentence, len(max_hierarchy)), -1, dtype=np.float32)

            hierarchy[:len(hid_array)] = hid_array + max_hierarchy
            document_id = np.full(max_sentence, -1, dtype=np.float32)
            document_id[:len(hid_array)] = it

            self.hierarchy.append(np.vstack([document_id, hierarchy.T]).T)
            max_hierarchy = max_hierarchy + np.max(hid_array, axis=0) + 1

            self.postits.append(
                t.postit
                if t.postit is not None
                else np.full(len(t.text[0]), -np.inf, dtype=np.float32)
            )

        self.scores = torch.tensor(np.array(self.scores)).to(dtype) * scale
        if thresh is not None:
            self.scores = (self.scores > thresh).to(dtype)

        self.genres = torch.vstack(self.genres).to(dtype)
        self.texts = torch.from_numpy(np.stack(self.texts)).to(dtype)
        self.hierarchy = torch.from_numpy(np.stack(self.hierarchy)).to(dtype)
        self.postits = torch.from_numpy(np.vstack(self.postits)).unsqueeze(1).to(dtype)
        self.masks = torch.from_numpy(np.stack(self.masks)).to(dtype)

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
        try:
            d = json.load(f)
        except:
            raise ValueError(f"Can't load book data {path}, possibly corrupted.")

    d["text"] = np.asarray(d["text"], dtype=np.float32)
    d["hierarchy_ids"] = np.asarray(d["hierarchy_ids"], dtype=np.int64)
    if d.get("postit") is not None:
        d["postit"] = np.asarray(d["postit"], dtype=np.float32)

    return d


class BookDataset:
    def __init__(
            self,
            train_path: List[str | Path] | str | Path,
            in_dim: int = 748,
            seq_len: int = 512,
            batch_size: int = 32,
            preload_size: int = 4,
            score_scale: float = 1.,
            score_thresh: float | None = None,
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
        if preload_size < self.batch_size:
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
        self.in_dim = in_dim
        self.score_scale = score_scale
        self.score_thresh = score_thresh
        self._doc_token: torch.Tensor | None = None

        self.in_order = in_order
        if self.in_order:
            self.idx = np.arange(len(self.book_entry_path_l))
        else:
            self.idx = np.random.permutation(len(self.book_entry_path_l))
        self.n_books = len(self.book_entry_path_l)
        self.pending_book_entries = []
        self.pool = ThreadPool(processes=self.n_cpu)
        self.epoch = 0
        self.current_idx = 0
        self.device = device
        self.preload_book_entries()

    @property
    def doc_token(self) -> torch.Tensor:
        if self._doc_token is None:
            self._doc_token = torch.finfo(torch.float16).min * torch.ones(self.in_dim)
        return self._doc_token

    def _reset(self):
        self.current_idx = 0
        self.epoch += 1
        if not self.in_order:
            np.random.shuffle(self.idx)

    def _get_book_entry(self, i):
        future = self.pending_book_entries.pop(i)
        try:
            return BookEntry(**future.get())
        finally:
            del future

    def preload_book_entries(self):
        while len(self.pending_book_entries) < self.preload_size:
            if self.current_idx >= len(self.book_entry_path_l):
                self._reset()
            self.pending_book_entries.insert(
                0,
                self.pool.apply_async(_load, (self.book_entry_path_l[self.idx[self.current_idx]],))
            )
            self.current_idx += 1

    def __len__(self):
        return len(self.book_entry_path_l) // self.batch_size

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.terminate()

    def __del__(self):
        pool = getattr(self, "pool", None)
        if pool is not None:
            pool.terminate()

    def __iter__(self):
        return self

    def __next__(self):
        results = []
        if self.in_order:
            for i_be in range(self.batch_size):
                if self.current_idx >= len(self.book_entry_path_l):
                    self._reset()
                results.append(self._get_book_entry(len(self.pending_book_entries) - 1 - i_be))
        else:
            while len(results) < self.batch_size:
                for i_be in range(len(self.pending_book_entries) - 1, -1, -1):
                    if self.pending_book_entries[i_be].ready():
                        results.append(self._get_book_entry(i_be))
                        if len(results) >= self.batch_size:
                            break
                time.sleep(0.0001)
        self.preload_book_entries()
        return BookBatch(
            results,
            scale=self.score_scale,
            thresh=self.score_thresh,
            seq_len=self.seq_len
        ).get_dict()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        return BookBatch(
            [_load(self.book_entry_path_l[self.idx[i]]) for i in idx],
            seq_len=self.seq_len
        ).get_dict()






