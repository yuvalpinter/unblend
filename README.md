# Will it Unblend?

This is the home for code and data from the paper [**Will it Unblend?**](https://www.aclweb.org/anthology/2020.findings-emnlp.138/), Findings of EMNLP, November 2020.

## Contents
**November 13**: We released the [complex words dataset](https://github.com/yuvalpinter/unblend/blob/main/complex_words.tsv) of 312 novel blends and compounds.
The data is in the following schema:
* **class**: whether the word is a blend or a compound.
* **word**: a word first appearing in the New York Times between November 2017 and March 2019 (taken from [NYTWIT](https://github.com/yuvalpinter/nytwit), follow link for details).
* **bases**: the words contributing to the complex word (space-delimited), manually annotated with help of originating NYT context.
* **sequence**: character-level annotation of the word reflecting each character's origin: **P**refix, **A**/**B**/**C** one of the bases (labeled successively according to their order in the *bases* column), **X** more than one base, **O** additional material, **S**uffix. See section 2 of the paper for details.
* **linearity**: whether the relation between the base-contributing parts of the word is linear: no **O**; no **A** preceded by a **B** or **X**; no **B** followed by an **A** or **X**; natural extension to words with a **C**. Compounds, by definition, contain no **X** or **O** and are always linear.
* **semantic relation**: the relationship between the *bases*, annotated according to the schema from [Tratz and Hovy, 2010](https://www.aclweb.org/anthology/P10-1070/).

Stay tuned for the following releases:
- [x] Code and data for reproducing the similarity experiments in section 3, including all BERT activations and lists of *smoothies*. (February 16, 2021)
- [ ] Code and data for reproducing the segmentation experiments in section 4.1, including models for the character LM, ~~[the character tagger](data/paxobs_charmanteau-to-nyt.pkl) and~~ the news-trained BPE table.
- [ ] Code and data for reproducing the recovery experiments in section 4.2~~, including [candidate](data/recovery_candidates_fasttext.tsv) [lists](data/recovery_candidates_glove840b.tsv)~~.

## Citing is Caring
Please use the following citation when you use our data or methods:
```
@inproceedings{pinter-etal-2020-will,
    title = "Will it Unblend?",
    author = "Pinter, Yuval  and
      Jacobs, Cassandra L.  and
      Eisenstein, Jacob",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.138",
    pages = "1525--1535",
}
```
