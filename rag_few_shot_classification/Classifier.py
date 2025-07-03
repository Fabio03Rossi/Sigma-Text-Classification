from transformers import pipeline


class Classifier:
    def __init__(self, model, tokenizer = None):
        self.model = model
        self.tokenizer = tokenizer

        if tokenizer is None:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model,
                device="cpu",
                use_fast=True
            )
        else:
            self.classifier = pipeline(
                "zero-shot-classification",
                model=model,
                device="cpu",
                use_fast=True,
                tokenizer=tokenizer
            )


    def classify(self, text, labels, multi_label, prompt: str = "si parla di {}"):
        result = self.classifier(text, labels, hypothesis_template=prompt, multi_label=True)
        return {label: score for label, score in zip(result["labels"], result["scores"])}


    @classmethod
    def parse_results(cls, classification, min_value):
        removals = []
        for entry in classification:
            if classification[entry] < min_value:
                removals.append(entry)
        for entry in removals:
            classification.pop(entry)
        return classification
