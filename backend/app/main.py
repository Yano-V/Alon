from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
from pathlib import Path
from app.engines.acoustic.processor import AcousticProcessor

app = FastAPI(title="Alon API Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUTS_DIR = Path("/Users/yanopalanas/Alon/outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "Welcome to Alon API Orchestrator", "status": "online"}

@app.post("/analyze")
async def analyze_speech(
    file: UploadFile = File(...), 
    speech_type: str = Form(...)
):
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    filename = f"recording_{file_id}{file_extension}"
    file_path = OUTPUTS_DIR / filename

    try:
        with file.file as buffer:
            with open(file_path, "wb") as buffer_file:
                shutil.copyfileobj(buffer, buffer_file)

        # --- ANALYSIS LAYER ---
        processor = AcousticProcessor(str(file_path))
        analysis_results = processor.analyze()

        return {
            "status": "success",
            "file_id": file_id,
            "speech_type": speech_type,
            "metrics": analysis_results
        }

    except Exception as e:
        # Cleanup file if analysis fails
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Analysis Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
