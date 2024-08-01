# Grammatical Error Correction using T5

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)
- [References](#references)

## Introduction
This project focuses on utilizing the T5 model from HuggingFace for grammatical error correction. T5 (Text-to-Text Transfer Transformer) is a versatile model that can convert all NLP tasks into a text-to-text format. In this project, we leverage T5 to identify and correct grammatical errors in sentences.

## Installation
 the necessary libraries installed.
```bash
pip install transformers datasets evaluate
pip install sentencepiece
pip install sacrebleu
````bash`


# Data Preparation

The dataset used in this project is loaded using the `datasets` library from HuggingFace. The dataset is preprocessed to fit the input requirements of the T5 model.

```python
from datasets import load_dataset
dataset = load_dataset('jfleg')

def preprocess_function(examples):
    inputs = examples["sentence"]
    targets = examples["corrected_sentence"]
    model_inputs = tokenizer(inputs, max_length=MAX_LENGTH, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LENGTH, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)
````python`

## 

