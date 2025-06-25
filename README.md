# NexText – A Smart NLP Platform for Summarization and Beyond

## Project Overview

NexText is an end-to-end Natural Language Processing (NLP) platform designed for advanced text summarization and extensible NLP tasks. Leveraging state-of-the-art transformer models, NexText enables users to generate concise and meaningful summaries from dialogues and documents, making it ideal for chat, meeting, and document summarization use cases.

## Key Features

- **State-of-the-art Summarization:** Utilizes the powerful [google/pegasus-cnn_dailymail](https://huggingface.co/google/pegasus-cnn_dailymail) transformer model for abstractive summarization.
- **Robust Dataset:** Trained and evaluated on the [knkarthick/samsum](https://huggingface.co/datasets/knkarthick/samsum) dataset, which contains real-world conversational data for dialogue summarization.
- **Modular Pipeline:** Includes configurable stages for data ingestion, validation, transformation, and model inference.
- **Transformation Model:** The platform preprocesses raw dialogues and summaries, tokenizes them using the Pegasus tokenizer, and transforms them into model-ready features. This transformation ensures compatibility with the model and optimizes performance for both training and inference.
- **Extensible Design:** Easily adaptable for other NLP tasks such as question answering, sentiment analysis, and more.

## How It Works

1. **Data Ingestion:** Downloads and extracts the SAMSum dataset, organizing it into train, validation, and test splits.
2. **Data Validation:** Ensures all required files are present and correctly formatted.
3. **Data Transformation:**
   - Loads the CSV files as Hugging Face datasets.
   - Applies the Pegasus tokenizer to both dialogues and summaries.
   - Converts the text data into input features (`input_ids`, `attention_mask`, `labels`) suitable for the transformer model.
   - Saves the processed dataset for efficient training and inference.
4. **Model Inference:** Uses the Pegasus model to generate abstractive summaries from new input dialogues.

## Technologies Used

- Python
- Hugging Face Transformers & Datasets
- PyTorch
- Google Pegasus Model
- SAMSum Dataset

## Workflows

1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update the configuration manager in src/config
5. Update the components
6. Update the pipeline
7. Update the main.py
8. Update the app.py

---

For more details, see the code and configuration files in this repository.
