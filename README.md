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
