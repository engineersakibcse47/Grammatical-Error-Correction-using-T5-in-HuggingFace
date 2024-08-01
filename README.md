# Grammatical Error Correction using T5 (Text-to-Text Transfer Transformer)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Tokenizer Selection](#tokenizer-selection)
- [Custom Data Preprocess Function](#custom-data-preprocess-function)
- [Training & Evaluation](#training-and-evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

## Introduction
This project focuses on utilizing the T5 model from HuggingFace for grammatical error correction. T5 (Text-to-Text Transfer Transformer) is a versatile model that can convert all NLP tasks into a text-to-text format. In this project, we leverage T5 to identify and correct grammatical errors in sentences.

## Installation
 the necessary libraries installed.
```
pip install transformers datasets evaluate
pip install sentencepiece
pip install sacrebleu
```


## Data Preparation
The data preparation involves loading and preprocessing the dataset. The dataset used for this project is `leslyarun/c4_200m_gec_train100k_test25k`.

```
from datasets import load_dataset

dataset_id = "leslyarun/c4_200m_gec_train100k_test25k"
dataset = load_dataset(dataset_id)
```

## Tokenizer Selection
I used the T5TokenizerFast tokenizer for efficient tokenization and encoding.

```
from transformers import T5TokenizerFast

model_id = "t5-small"
tokenizer = T5TokenizerFast.from_pretrained(model_id)
```

## Custom Data Preprocess Function
A custom preprocessing function is defined to prepare the inputs and targets for the model. The function tokenizes the input and target texts and truncates them to a maximum length of 128 tokens.

```
MAX_LENGTH = 128

def preprocess_function(examples):
    inputs = [example for example in examples['input']]
    targets = [example for example in examples['output']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=MAX_LENGTH, truncation=True)
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
```

## Model Initialization
The T5-small model is loaded, and a data collator is defined to prepare the data for training.

```
from transformers import TFAutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

model = TFAutoModelForSeq2SeqLM.from_pretrained(model_id)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, return_tensors="tf")
```

## Training and Evaluation
The model is trained on the prepared dataset, and various evaluation metrics are used to assess its performance. The BLEU score is a key metric for evaluating the quality of the generated corrections.

## Conclusion
This project demonstrates the implementation of a Grammatical Error Correction system using the T5 model in HuggingFace. The steps include data preparation, tokenizer selection, model initialization, and training. The system aims to improve the grammatical accuracy of input texts, and its performance is evaluated using relevant metrics.

## Future Work
Future improvements could include experimenting with larger T5 models, fine-tuning on more extensive datasets, and incorporating additional evaluation metrics to further enhance the system's accuracy and robustness.








