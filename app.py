from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from TextSummariser.components.prediction import PredictionPipeline, PredictionConfig
from pathlib import Path


text:str = "What is Text Summarization?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(text: str):
    try:
        # Initialize prediction pipeline with proper config
        config = PredictionConfig(
            model_path=Path("artifacts/model_trainer/pegasus-samsum-model"),
            tokenizer_path=Path("artifacts/model_trainer/tokenizer"),
            max_length=128,
            min_length=10,
            num_beams=4
        )
        obj = PredictionPipeline(config)
        summary = obj.predict(text)
        return {"summary": summary, "original_text": text}
    except Exception as e:
        return {"error": str(e)}
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)