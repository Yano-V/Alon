# Alon - Automated Web-Based Audio Speech Evaluator

## Project Structure
- `frontend/`: Next.js 14 application for the User Interface.
- `backend/`: FastAPI orchestrator and the Analysis Layer.
    - `app/engines/acoustic/`: Librosa-based feature extraction (Jitter, Shimmer, etc.).
    - `app/engines/linguistic/`: Transcription and filler word detection.
    - `app/engines/generative/`: LLM-powered coaching recommendations.
- `ground_truth/`: Reference data and Praat exports for validation.
- `outputs/`: Local storage for generated heatmaps and transcripts.

## Setup Guide

### Backend
1. Navigate to `backend/`
2. Install dependencies: `pip install -r requirements.txt`
3. Rename `.env.example` to `.env` and add your API keys.
4. Start server: `python app/main.py`

### Frontend
1. Navigate to `frontend/`
2. Initialize Next.js: `npx create-next-app@latest . --typescript --tailwind --eslint`
3. Run development server: `npm run dev`
