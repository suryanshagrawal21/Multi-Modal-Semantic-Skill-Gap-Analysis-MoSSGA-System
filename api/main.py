from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import shutil

# Import the existing MoSSGA pipeline and parsers
from src.mossga_engine import run_mossga_pipeline
from src.parser import extract_text

app = FastAPI(title="MoSSGA AI API", version="1.0.0")

# Enable CORS for the React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    github_url: str = ""
    target_role: str = ""
    job_description: str = ""
    manual_skills: str = ""

@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "MoSSGA Hybrid AI Engine running."}

@app.post("/api/analyze")
async def analyze_skills(
    github_url: str = Form(""),
    target_role: str = Form(""),
    job_description: str = Form(""),
    manual_skills: str = Form(""),
    files: list[UploadFile] = File(None)
):
    """
    Endpoint to process multi-modal inputs and return the complete MoSSGA hybrid report.
    """
    resume_text_combined = ""

    # Process uploaded resumes
    if files:
        for file in files:
            if file.filename:
                # Save temporarily
                ext = file.filename.rsplit(".", 1)[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    tmp_path = tmp.name

                text = extract_text(tmp_path)
                os.remove(tmp_path)
                if text:
                    resume_text_combined += "\n" + text

    # Also append manual skills to resume text so the extractor picks them up
    if manual_skills:
        resume_text_combined += "\n" + manual_skills 

    if not resume_text_combined and not github_url:
        raise HTTPException(status_code=400, detail="Must provide at least a resume, manual skills, or GitHub URL.")

    # Run the core MoSSGA engine
    try:
        report = run_mossga_pipeline(
            resume_text=resume_text_combined,
            github_input=github_url,
            jd_text=job_description,
            target_role=target_role
        )
        return {"success": True, "data": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
