import sys
import argparse
import json
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
nltk.download('punkt_tab')


from inkler.constants import GENRES, GENRE_DICT
def parse_args(args):
    parser = argparse.ArgumentParser("Parse txt documents to hierarchical document structure")
    parser.add_argument("--path", required=True, type=str,
                        help="Path to the txt file")
    parser.add_argument("--out", required=True, type=str,
                        help="Safe parsed document to")
    parser.add_argument("--title", required=True, type=str,
                        help="Book title")
    parser.add_argument("--author", required=True, type=str,
                        help="Author name")
    parser.add_argument("--score", required=True, type=float,
                        help="Quality score")
    parser.add_argument("--genre", nargs="+", type=str, required=True, choices=GENRES,
                        help="Genre list")
    parser.add_argument("--postit", type=str,
                        help="If passed also load postit at path")
    parser.add_argument("--head", type=int, default=0,
                        help="Number of lines at the beginning to skip")
    parser.add_argument("--tokenizer", type=str,
                        help="If passed, sentence transformer to be used")

    return parser.parse_args(args)


def is_empty(line):
    return len(line.strip("\n").strip(" ")) < 1


def strip_line(line):
    return line.strip("\n").lstrip(' ').replace("\n", " ")


def main(args):
    path = args.path
    out = args.out
    title = args.title
    author = args.author
    score = args.score
    genre = args.genre
    n_head = args.head
    postit_path = args.postit
    tokenizer = args.tokenizer

    embedded_doc = []
    last_sentence = []
    paragraph_id = []
    chapter_id = []
    with open(path, mode="r", encoding='utf-8-sig') as file:
        for i_line, line in enumerate(file):
            if i_line == n_head:
                break
            if is_empty(line):
                continue
            last_sentence.append(strip_line(line))

        if len(last_sentence) > 0:
            combined_string = " ".join(last_sentence)
            embedded_doc.extend([s + "." for s in combined_string.split(". ")])
            paragraph_id.extend([0] * len(embedded_doc))
            chapter_id.extend([0] * len(embedded_doc))

        paragraph_cntr = 1
        chapter_cntr = 1
        main_text = nltk.sent_tokenize(file.read())
        for sentence in main_text:
            stripped = strip_line(sentence)
            if stripped.lower().startswith("chapter"):
                chapter_cntr += 1
            if "\n\n" in sentence:
                split_sentences = sentence.split("\n\n")
                for s in split_sentences:
                    if is_empty(s):
                        continue
                    stripped = strip_line(s)
                    embedded_doc.append(stripped)
                    paragraph_id.append(paragraph_cntr)
                    chapter_id.append(chapter_cntr)
                    paragraph_cntr += 1
                continue

            if is_empty(stripped):
                continue
            embedded_doc.append(stripped)
            paragraph_id.append(paragraph_cntr)
            chapter_id.append(chapter_cntr)
            if sentence.endswith("\n\n"):
                paragraph_cntr += 1

    sentence_embedder = None
    if tokenizer is not None:
        sentence_embedder = SentenceTransformer(tokenizer, cache_folder=".tokenizer/")
        for i_s, sentence in tqdm(enumerate(embedded_doc), total=len(embedded_doc), desc="Processing sentences"):
            embedded_doc[i_s] = sentence_embedder.encode(
                sentences=sentence,
                convert_to_tensor=False,
                show_progress_bar=False
            )

    embedded_doc = np.vstack(embedded_doc)
    hierarchy_ids = np.vstack([chapter_id, paragraph_id]).T
    postit = None
    if postit_path is not None:
        with open(postit_path, mode="r", encoding='utf-8') as f:
            postit = f.read()
        if sentence_embedder is not None:
            postit = sentence_embedder.encode(sentences=postit, convert_to_tensor=False)

    genre_ids = []
    for g in genre:
        genre_ids.append(GENRE_DICT[g])

    data_dict = {
        "author": author,
        "title": title,
        "score": score,
        "genre": genre_ids,
        "text": embedded_doc.tolist(),
        "hierarchy_ids": hierarchy_ids.tolist(),
        "postit": postit.tolist() if postit is not None else None,
    }

    with open(out, "w") as parsed_file:
        json.dump(data_dict, parsed_file)

if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))


