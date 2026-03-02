import sys
import os
import argparse
import json
import warnings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
import pandas as pd
nltk.download('punkt_tab')

from inkler.constants import GENRE_DICT, SENTENCE_MODEL
from inkler.utils import roman_to_int


def parse_args(args):
    parser = argparse.ArgumentParser("Parse txt documents to hierarchical document structure")
    parser.add_argument("--meta_path", default="../data/inkling.csv", type=str,
                        help="Path to meta table")
    parser.add_argument("--book_dir", default="../data/raw/", type=str,
                        help="Book directory")
    parser.add_argument("--out_dir", default="../data/parsed/", type=str,
                        help="Output directory")

    return parser.parse_args(args)


def is_empty(line):
    return len(line.strip("\n").strip(" ")) < 1


def strip_line(line):
    return line.strip("\n").lstrip(' ').replace("\n", " ")


def main(args):
    meta_path = args.meta_path
    book_dir = args.book_dir
    out_dir = args.out_dir
    sentence_embedder = SentenceTransformer(SENTENCE_MODEL, cache_folder=".tokenizer/")

    meta_df = pd.read_csv(meta_path, index_col=None)
    for _, row in meta_df.iterrows():
        last_name = row["Author, Last"].lower()
        if not row.isna()["Author, First"]:
            first_name = row["Author, First"].lower().replace(" ", "").replace(".", "")
        else:
            first_name = ""
        title = row["Title"]
        author = f"{row["Author, First"]} {row["Author, Last"]}"
        score = float(row["Score Leo"])
        genre = row["Genre"].split("; ")
        postit_str = row["Postit"]
        n_head = row["Head"]
        if row.isna()["Postit"] or postit_str is None or len(postit_str) == 0:
            postit_str = None

        print(f"Start processing {title} by {author}")
        file_name = f"{last_name} {first_name}_{title.lower()}"
        path = os.path.abspath(os.path.join(book_dir, f"{file_name}.txt"))
        if not os.path.isfile(path):
            warnings.warn(f"Book {path} not found.")
            continue
        out = os.path.abspath(os.path.join(out_dir, f"{file_name}.json"))
        if os.path.isfile(out):
            print("\tBook already processed. Skip")
            continue
        embedded_doc = []
        last_sentence = []
        paragraph_id = []
        chapter_id = []
        with open(path, mode="r", encoding='utf-8-sig') as file:
            for i_line, line in enumerate(file):
                if "*** START OF THE PROJECT GUTENBERG EBOOK" in line:
                    break
            n_line = i_line
            for i_line, line in enumerate(file):
                if n_line + i_line >= n_head:
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
            chapter_cntr = 0
            main_text = nltk.sent_tokenize(file.read())
            for sentence in main_text:
                stripped = strip_line(sentence)
                if "*** END OF THE PROJECT GUTENBERG EBOOK" in stripped:
                    break
                if is_empty(sentence):
                    continue
                if stripped.lower().startswith("chapter") or stripped.lower().startswith("kapitel") or stripped.lower().startswith("chapitre") or roman_to_int(stripped.upper().replace(".", "")) is not None:
                    chapter_cntr += 1
                    split_list = sentence.split("\n\n")
                    split_list = [strip_line(s) for s in split_list if not is_empty(strip_line(s))]
                    if len(split_list) > 1:
                        striped = strip_line(" ".join(split_list[1:]))
                    else:
                        continue
                elif "\n\n" in sentence:
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

        for i_s, sentence in tqdm(enumerate(embedded_doc), total=len(embedded_doc), desc="Processing sentences"):
            embedded_doc[i_s] = sentence_embedder.encode(
                sentences=sentence,
                convert_to_tensor=False,
                show_progress_bar=False
            )

        embedded_doc = np.vstack(embedded_doc)
        hierarchy_ids = np.vstack([chapter_id, paragraph_id]).T
        postit = None
        if postit_str is not None:
            postit = sentence_embedder.encode(sentences=postit_str, convert_to_tensor=False)

        genre_ids = []
        for g in genre:
            if g.lower() not in GENRE_DICT:
                warnings.warn(f"{g} not found in genre list. Skip")
                continue
            genre_ids.append(GENRE_DICT[g.lower()])

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

        print(f"\tBook {title} by {author} successfully processed.")


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))


