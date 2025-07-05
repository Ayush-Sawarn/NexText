from transformers import AutoTokenizer
from datasets import load_dataset
import os
from TextSummariser.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):

        # Ensure input is always a list of strings
        dialogues = example_batch['dialogue']
        summaries = example_batch['summary']

        # If a single string sneaks in, wrap it in a list
        if isinstance(dialogues, str):
            dialogues = [dialogues]
        if isinstance(summaries, str):
            summaries = [summaries]

        # Remove None values (if any)
        dialogues = [d if d is not None else "" for d in dialogues]
        summaries = [s if s is not None else "" for s in summaries]

        input_encodings = self.tokenizer(
            dialogues, max_length=1024, truncation=True, padding="max_length"
        )
        target_encodings = self.tokenizer(
            summaries, max_length=128, truncation=True, padding="max_length"
        )
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_dataset(
            "csv",
            data_files={
                "train": "artifacts/data_ingestion/samsum_dataset/train.csv",
                "validation": "artifacts/data_ingestion/samsum_dataset/validation.csv",
                "test": "artifacts/data_ingestion/samsum_dataset/test.csv"
            }
        )
        print(dataset_samsum["train"].column_names)  # Debug: check column names
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched=True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir, "samsum_dataset"))
