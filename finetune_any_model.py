# -*- coding: utf-8 -*-
"""finetune_any_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1v2mzAwx4HDTV8S7FR7oI3KL7U5XX3VtF
"""

# !pip install accelerate transformers peft bitsandbytes datasets

from datasets import load_dataset, Dataset
import torch

from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,BitsAndBytesConfig,DataCollatorForLanguageModeling
    )

from peft import LoraConfig, get_peft_model, PeftModel

import pandas as pd

class LoRAFineTuner:
    def __init__(self,model_name,dataset_name,output_dir):
      """
      this is initlization of the class parameter
      """
      print("params initlized")
      self.model_name=model_name
      self.dataset_name=dataset_name
      self.output_dir=output_dir
      self.tokenizer=None
      self.model=None
      self.tokenized_data=None

    def load_tokenizer(self):
      """
      this function to define the tokenizer of the model
      """
      self.tokenizer=AutoTokenizer.from_pretrained(self.model_name,trust_remote_code=True)
      self.tokenizer.pad_token=self.tokenizer.eos_token

    def load_model(self):
      """
      this function to define the model of the model
      """
      bnb_config=BitsAndBytesConfig(
          load_in_4bit=True,
          bnb_4bit_use_double_quant=True,
          bnb_4bit_quant_type="nf4",
          bnb_4bit_compute_dtype=torch.float16
      )

      # quantizatied model
      self.model=AutoModelForCausalLM.from_pretrained(
          self.model_name,
          device_map={"": 0},
          trust_remote_code=True,
          quantization_config=bnb_config
      )

    def apply_lora(self):
      """
      this function to define the lora model of the model
      """
      config=LoraConfig(
          r=16,
          lora_alpha=32,
          target_modules=["q_proj", "v_proj"],
          lora_dropout=0.05,
          bias="none",
          task_type="CAUSAL_LM"
      )
      # applied lora on quantizatied model
      self.model=get_peft_model(self.model,config)

      self.model.print_trainable_parameters()

    def load_and_tokenize_dataset(self):
      """
      this function will load the and it will perform the tokenization on it
      """
      data=load_dataset(self.dataset_name,'main',split="train")

      data_df = data.to_pandas()
      print(data_df.head())

      text_column = data_df.columns[0]  # Select first column if unsure
      print(text_column)

      if "question" in data_df.columns and "answer" in data_df.columns:
            data_df["text"] = data_df.apply(lambda x: f"question: {x['question']} answer: {x['answer']}", axis=1)
      else:
            data_df["text"] = data_df[text_column]

      # Convert back to Hugging Face dataset
      data = Dataset.from_pandas(data_df)

      # Tokenize dataset
      def tokenize(sample):
        return self.tokenizer(sample["text"], padding=True, truncation=True, max_length=512)

      self.tokenized_data = data.map(tokenize, batched=True, desc="Tokenizing data", remove_columns=data.column_names)

    def train(self,epochs: int = 1, batch_size: int = 4, learning_rate: float = 2e-4, max_steps: int = 1000):
      """
      this function will perform the training
      """
      training_args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            save_strategy="epoch",
            logging_steps=100,
            max_steps=max_steps,
            num_train_epochs=epochs,
            push_to_hub=True,
            report_to="none"

        )

      trainer = Trainer(
          model=self.model,
          train_dataset=self.tokenized_data,
          args=training_args,
          data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
      )

      trainer.train()

    def save_model(self,model_repo:str):
      """
      this function will save the model
      """
      base_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch.float32)
      peft_model = PeftModel.from_pretrained(base_model, self.output_dir, from_transformers=True)
      merged_model = peft_model.merge_and_unload()

      merged_model.push_to_hub(model_repo)

      print("saving the model")

    def run(self):
      """
      this function will run the whole process
      """
      print("starting finetunine process")
      self.load_tokenizer()
      print("tokenizer loaded")

      self.load_model()
      print("model loaded")

      self.apply_lora()
      print("lora applied")

      self.load_and_tokenize_dataset()
      print("dataset loaded and tokenized")

      self.train()
      print("model trained")

      self.save_model()
      print("model saved")

###userThe ratio of boys to girls in a math ###assistantIf the ratio of boys to girls in the math class is 5:8, the total ratio of the numb

model_name="microsoft/phi-1_5"
dataset_name="gsm8k"
output_dir="phi-1_5-finetuned"

fine_tuner=LoRAFineTuner(model_name,dataset_name,output_dir)

fine_tuner.run()

data=load_dataset("gsm8k",'main',split="train")

data_df = data.to_pandas()

print(data_df.head())

data_df.columns

data_df.columns[0]

text_column = data_df.columns[0]

print(text_column)

if "question" in data_df.columns and "answer" in data_df.columns:
      data_df["text"] = data_df.apply(lambda x: f"question: {x['question']} answer: {x['answer']}", axis=1)
else:
      data_df["text"] = data_df[text_column]

data_df["text"][0]

assisgnment:

framework: 1.unsloth 2.llama-factory

models:
a. mistral
b. llama
c. phi

pefrom a finetuning using lora and qlora

but this time you suppose to use framework(1.unsloth 2.llama-factory)