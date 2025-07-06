import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from TextSummariser.components.prediction_demo import PredictionPipeline, PredictionConfig
from pathlib import Path

def test_summarization():
    """Test the trained model with a sample text."""
    
    # Configuration for prediction
    config = PredictionConfig(
        model_path=Path("artifacts/model_trainer/pegasus-samsum-model"),
        tokenizer_path=Path("artifacts/model_trainer/tokenizer"),
        max_length=128,
        min_length=10,
        num_beams=4
    )
    
    # Initialize prediction pipeline
    print("Loading trained model...")
    predictor = PredictionPipeline(config)
    print("Model loaded successfully!")
    
    # Sample dialogue text (similar to SAMSum format)
    sample_text = """
    Person A: Hey, how was your day?
    Person B: It was pretty good! I had a meeting with the team about the new project.
    Person A: Oh, how did that go?
    Person B: Really well! We discussed the timeline and everyone seems to be on the same page.
    Person A: That's great! Any challenges?
    Person B: Not really, just need to finalize the budget by Friday.
    Person A: Got it. Well, good luck with that!
    Person B: Thanks! How about you?
    Person A: Pretty busy with deadlines, but managing.
    Person B: Hang in there! You've got this.
    """
    
    print("\n" + "="*50)
    print("INPUT TEXT:")
    print("="*50)
    print(sample_text.strip())
    
    # Generate summary
    print("\n" + "="*50)
    print("GENERATED SUMMARY:")
    print("="*50)
    summary = predictor.predict(sample_text)
    print(summary)
    
    # Test with another example
    print("\n" + "="*50)
    print("ANOTHER EXAMPLE:")
    print("="*50)
    
    another_text = """
    Person A: Did you finish the report?
    Person B: Almost done, just need to add the final section.
    Person A: When can you send it?
    Person B: I'll have it ready by 3 PM today.
    Person A: Perfect, thanks!
    """
    
    print("Input:", another_text.strip())
    print("Summary:", predictor.predict(another_text))

if __name__ == "__main__":
    test_summarization() 