# Deployment Guide: MoSSGA Hybrid System

Because this application is a modern, modular, production-ready system consisting of a **React Frontend** and a **FastAPI Backend**, deployment requires two steps. We recommend **Render.com** for the backend and **Vercel** for the frontend.

## Step 1: Deploy the Backend (FastAPI) on Render
Render is perfect for Python APIs.

1. Go to [Render.com](https://render.com) and create a new **Web Service**.
2. Connect your GitHub repository: `Intelligent_AI_Resume_Screening_System`.
3. Fill out the deployment details:
   - **Name**: `mossga-api` (or whatever you prefer).
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Click **Create Web Service**. 
5. Wait for the build to finish. Once it says "Live", copy the resulting URL (e.g., `https://mossga-api.onrender.com`).

---

## Step 2: Deploy the Frontend (React) on Vercel
Vercel is the easiest global CDN for modern React applications.

1. Go to [Vercel.com](https://vercel.com) and create a **New Project**.
2. Import the `Intelligent_AI_Resume_Screening_System` repository from GitHub.
3. **CRITICAL STEP: Configure the Project**:
   - In the **Root Directory** section, click Edit and select `frontend`.
   - The Framework Preset should automatically detect "Vite".
4. **Environment Variables**:
   - Add a new variable named `VITE_API_URL`.
   - For the Value, paste the URL you got from Render in Step 1 (e.g., `https://mossga-api.onrender.com`). **Do not put a trailing slash!**
5. Click **Deploy**.

## Conclusion
Once Vercel finishes deploying, they will provide you the final live URL! Your professional frontend will be hosted on Vercel, securely calling your heavy AI calculations over on Render.
