import os
import pandas as pd
import torch
from datasets import Dataset
from setfit import SetFitModel, TrainingArguments, Trainer, get_templated_dataset, sample_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-7f4915f4-0bb6-f23e-021c-200070c6cf70"
os.environ["WANDB_DISABLED"] = "true"

state = ["Guasto", "Funzionante"]

df_training = pd.read_excel("docs/training_docs/ClosingNotes-sample-20250610-training-state.xlsx")
df_validation = pd.read_excel("docs/training_docs/ClosingNotes-sample-20250610-validation.xlsx")
df_test = pd.read_excel("docs/training_docs/ClosingNotes-sample-20250617-test.xlsx")

# Rimappa le colonne relative ad ogni elemento della lista in una colonna "labels" e le Closing Notes in "text"
# Inoltre ripulisce le caselle vuote sostituendole con una stringa "Empty"
def remap_dataset(data, features):
    #data = data.map(lambda entry: {"labels": [entry[label] for label in features]})
    dictionary = {"text": data["Closing Note"], "labels": data['State']}
    data = Dataset.from_dict(dictionary)
    data = Dataset.from_pandas(data.to_pandas().dropna())
    data = Dataset.from_pandas(data.to_pandas().replace(r'^\s*$', "Empty", regex=True))
    return data

def main():
    dataset = Dataset.from_pandas(df_training.iloc[:, 9:18])
    train_dataset = remap_dataset(dataset, state)

    train_dataset = sample_dataset(train_dataset, label_column="labels")
    train_dataset = get_templated_dataset(train_dataset, candidate_labels=state, sample_size=10, template="L'ATM è stato trovato {}", label_column="labels")

    params = {"device": torch.device("cuda"), 'out_features': 2, 'temperature': 0.5}
    model = SetFitModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", params=params)

    args = TrainingArguments(
        # Optional training parameters:
        body_learning_rate=1.9959376752033417e-05,
        num_epochs=8,
        batch_size=12,
        warmup_proportion=0.1,
        sampling_strategy="unique",
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        run_name="finetune-setfit",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        column_mapping={"text": "text", "labels": "label"},
    )

    trainer.train()

    model = trainer.model
    model.save_pretrained("hello/stateModel3")

if __name__ == "__main__":
    main()