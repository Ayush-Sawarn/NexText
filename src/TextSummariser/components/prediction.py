import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PredictionConfig:
    model_path: Path
    tokenizer_path: Path
    max_length: int = 128
    min_length: int = 10
    num_beams: int = 4


class PredictionPipeline:
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load the trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path).to(self.device)
        
    def predict(self, text: str) -> str:
        """
        Generate a summary for the given text.
        
        Args:
            text (str): Input text to summarize
            
        Returns:
            str: Generated summary
        """
        # Tokenize the input text
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=self.config.max_length,
                min_length=self.config.min_length,
                num_beams=self.config.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode the generated summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return summary
    
    def predict_batch(self, texts: list) -> list:
        """
        Generate summaries for a batch of texts.
        
        Args:
            texts (list): List of input texts to summarize
            
        Returns:
            list: List of generated summaries
        """
        summaries = []
        for text in texts:
            summary = self.predict(text)
            summaries.append(summary)
        
        return summaries 