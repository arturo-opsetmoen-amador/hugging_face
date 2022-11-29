#!/usr/bin/python

"""
Transformers, what can they do?
HuggingFace course
Chapter 1, exercise 4
"""

from transformers import BertTokenizer, AutoTokenizer


tokenized_text = "Jim Henson was a puppeteer".split()


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
auto_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

print(tokenized_text)
print(bert_tokenizer(tokenized_text))
print(auto_tokenizer(tokenized_text))