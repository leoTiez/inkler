# Inkling

`Inkling` is a method for generating font patterns based on trained hierarchical texts, such as books. The entire book, individual chapters, paragraphs, or single sentences are uniformly represented by a graphic pattern. It is an artistic research project to analyse scripture generation, and the encoding capacity of glyphs with a limited number of strokes.

## Pre-processing
Texts are pre-processed per sentence, and each sentence is represented in a single vector using the `paraphrase-multilingual-MiniLM-L12-v2` model. Sentences are then associated to paragraphs, chapters, and books unsing indices. The correct association to the hierarchy level with incrementing indices without gaps is essential for adequate execution.

## The Inkler
The `Inkler` can be understood as a graph convolution network with attention blocks. It consists of an encoder and a decoder part. The vector sequence is analysed using a cross-correlation attention (XCA) or a conventional attention mechanism. The output is then pooled with respect to the provided hierarchy indices. Pooling can be mean, sum, argmax or argmin. Default is argmax. The final vector is the document representation. The document can be read several times with the document vector of the previous reading appended. This allows the hierarchies to communicate with each other. Default is parsing the text twice, the second time permitting informing the attention mechanism of the higher hierarchical structure.
In the latent space, different glyph modules can be applied to generate symbols. Those generate a new symbol based on a provided number of strokes, following, or at least being constrained by, a prescribed stroke topology. A soft interpolation noise pattern is applied before reconstructing the original vectors back based on the symbols. Note that symbols are generated for every hierarchy, i.e. sentences, paragraphs, chapters, and the entire book. The latent space can be additionally constrained with genre, scoring, or external review information.
The reconstructed vector is then demultiplexed back to the original structure. The aim is to recover the lower hierarchy vector sequence based on the pooled single higher hierarchy vector. 
All together, it is a multi-loss objective that aims to represent as much contextual information about a book as possible in a human readable glyph.

<p align="center">
  <a href="https://<user>.github.io/<repo>/">
    <img src="animations/inkler.svg" width="640" alt="Inkler: hierarchical pooling, glyph generation and demultiplexing">
  </a>
  <br>
  <sub>Click for the interactive version with pause and scrubbing.</sub>
</p>

## Installation
Install the `Inkler` by running
```commandline
python3 -m venv .inkler
source .inkler/bin/activate
python3 -m pip install -e .
```
The code was implemented and tested using Python 3.13 and minimum 3.9 is recommended.