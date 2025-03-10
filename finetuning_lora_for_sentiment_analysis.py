# -*- coding: utf-8 -*-
"""FineTuning_LoRA_for_sentiment_analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Skcr-I3Nb3G94Djf-K1r9U5PGCcY0Vav

<a href="https://colab.research.google.com/github/fshnkarimi/Fine-tuning-an-LLM-using-LoRA/blob/main/FineTuning_LoRA.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""

!pip install datasets
!pip install transformers
!pip install peft
!pip install evaluate

from datasets import load_dataset, DatasetDict, Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer)

from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np

"""# dataset"""

# sst2
# The Stanford Sentiment Treebank consists of sentences from movie reviews and human annotations of their sentiment. The task is to predict the sentiment of a given sentence. It uses the two-way (positive/negative) class split, with only sentence-level labels.
dataset = load_dataset("glue", "sst2")
dataset

# len(dataset['train']['label'])
# np.array(dataset['train']['label'])
np.array(dataset['train']['label']).sum()

# display % of training data with label=1
np.array(dataset['train']['label']).sum()/len(dataset['train']['label'])

"""# model"""

model_checkpoint = 'roberta-base'

# define label maps
id2label = {0: "Negative", 1: "Positive"}
label2id = {"Negative":0, "Positive":1}

# generate classification model from model_checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id)

# display architecture
model

"""# preprocess data"""

# create tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

# add pad token if none exists
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

# create tokenize function
def tokenize_function(examples):
    # extract text
    text = examples["sentence"]

    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs

# tokenize training and validation datasets
tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset

# create data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

"""# evaluation"""

# import accuracy evaluation metric
accuracy = evaluate.load("accuracy")

# define an evaluation function to pass into trainer later
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)

    return {"accuracy": accuracy.compute(predictions=predictions, references=labels)}

"""# Apply untrained model to text"""

# define list of examples
text_list = ["a feel-good picture in the best sense of the term .", "resourceful and ingenious entertainment .", "it 's just incredibly dull .", "the movie 's biggest offense is its complete and utter lack of tension .",
             "impresses you with its open-endedness and surprises .", "unless you are in dire need of a diesel fix , there is no real reason to see it ."]

print("Untrained model predictions:")
print("----------------------------")
for text in text_list:
    # tokenize text
    inputs = tokenizer.encode(text, return_tensors="pt")
    # compute logits
    logits = model(inputs).logits
    # convert logits to label
    predictions = torch.argmax(logits)

    print(text + " - " + id2label[predictions.tolist()])

"""# Train model"""

peft_config = LoraConfig(task_type="SEQ_CLS",
                        r=4,
                        lora_alpha=32,
                        lora_dropout=0.01,
                        target_modules = ['query'])

peft_config

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# hyperparameters
lr = 1e-3
batch_size = 16
num_epochs = 5

# define training arguments
training_args = TrainingArguments(
    output_dir= model_checkpoint + "-lora-text-classification",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=num_epochs,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# train model
trainer.train()

"""# Generate prediction"""

model.to('cpu')

print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to("cpu")

    logits = model(inputs).logits
    predictions = torch.max(logits,1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])

model.save_pretrained("sunj_finetuned_roberta-base-lora-text-classification")

import shutil

# Model folder ka naam jo tu save kar chuka hai
model_folder = "/content/sunj_finetuned_roberta-base-lora-text-classification"

# Model ko zip me convert karo
shutil.make_archive("sunj_finetuned_roberta-base-lora-text-classification", 'zip', model_folder)

print("Model successfully zipped as sunj_finetuned_roberta-base-lora-text-classification.zip")

from google.colab import files

# ZIP file ko download karne ke liye
files.download("sunj_finetuned_roberta-base-lora-text-classification.zip")

import zipfile

# ZIP file ka path
zip_path = "sunj_finetuned_roberta-base-lora-text-classification.zip"

# Extract karne ke liye
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("./extracted_model")

print("Model successfully extracted!")

tokenizer.save_pretrained("tokenizer")

import json

config=json.load(open("/content/extracted_model/adapter_config.json"))

! pip install huggingface_hub

from huggingface_hub import notebook_login

notebook_login()

from transformers import AutoModelForTokenClassification, AutoTokenizer
from huggingface_hub import HfApi

# Model directory (jisme fine-tuned model hai)
model_dir = "/content/extracted_model"

model_fine_tuned=AutoModelForSequenceClassification.from_pretrained("/content/extracted_model")
# Push model to Hugging Face
model_fine_tuned.push_to_hub("sunjupskilling/sunj-finetuned-roberta-base-lora-text-classification")

