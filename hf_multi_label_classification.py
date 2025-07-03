import os

import pandas as pd
import torch
from datasets import Dataset
from setfit import get_templated_dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding

os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def preprocess_function(example):
    text = f"{example['text']}"
    all_labels = features
    labels = [0. for _ in range(len(dataset))]
    for label in all_labels:
        label_id = class2id[label]
        labels[label_id] = 1.

    example = tokenizer(text, truncation=True)
    example['labels'] = labels
    return example

model_path = 'microsoft/deberta-v3-small'
tokenizer = AutoTokenizer.from_pretrained(model_path)

df = pd.read_excel("docs/LabelledSamplesShort.xlsx")
dataset = Dataset.from_pandas(df)

features = ["UNK", "CT", "NV", "NE", "NF", "SHUTTER", "CASSETTE", "CRM9250"]

class2id = {class_:id for id, class_ in enumerate(features)}
id2class = {id:class_ for class_, id in class2id.items()}

tokenized_dataset = dataset.map(preprocess_function)

train_dataset = get_templated_dataset(dataset, candidate_labels=features, sample_size=8, label_column="labels")

print(tokenized_dataset)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(
   model_path, num_labels=len(features),
    id2label=id2class, label2id=class2id,
    problem_type = "multi_label_classification"
)

torch.cuda.empty_cache()

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

from scikit_ollama import print_and_w

preds = model.predict(["""Riscontrato apparato con fs 9301. Aperto dispensatore e rinvenuto biglietto di carta, scritto a penna, incastrato all'interno. Rimosso corpo estrano e resettato dspositivo. Carta stampante mancante, verrà la vigilanza in un secondo momento a sostituirla.""",
"""Rimosse due banconote accartocciate. Controllo percorso banconote gruppo ricircolo con esito positivo.""",
"""Guasto riscontrato : cinghia escrow fuori sede. Riposizionato cinghia,  collaudo con esito positivo"""
])

with open("hf.txt", 'a') as f:
    print_and_w(f, preds)

