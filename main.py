import os, shutil, joblib, cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse 
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pose_detect import process_image, process_frame_raw, webcam_generator

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

os.makedirs("uploads", exist_ok=True)

# Note: Ensure your API key is active and has Gemini 2.0 Flash access
client = genai.Client(api_key="AIzaSyDtSl3AlVk3nTdHtORcJ_s5pG8s0k3eUCo")

model = joblib.load("outcome_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Constant feature order used during training
FEATURE_ORDER = ["elbow_angle", "arm_extension", "hip_rotation", "torso_rotation"]

@app.get("/", response_class=HTMLResponse)
async def home(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})

@app.post("/predict", response_class=HTMLResponse)
async def predict(req: Request, image: UploadFile = File(...)):
    path = f"uploads/{image.filename}"
    with open(path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    original, skeleton, feats = process_image(path)

    if feats is None:
        return templates.TemplateResponse("index.html", {
            "request": req,
            "prediction": "No valid batting pose detected"
        })

    orig_path = f"/uploads/original_{image.filename}"
    skel_path = f"/uploads/skeleton_{image.filename}"
    cv2.imwrite(orig_path[1:], original)
    cv2.imwrite(skel_path[1:], skeleton)
    
    # Corrected: Use FEATURE_ORDER to ensure scalar transform matches model training
    X = scaler.transform([[feats[f] for f in FEATURE_ORDER]])
    probs = model.predict_proba(X)[0]

    prediction = le.inverse_transform([probs.argmax()])[0]
    confidence = round(max(probs) * 100, 2)

    return templates.TemplateResponse("index.html", {
        "request": req,
        "prediction": prediction,
        "confidence": confidence,
        "features": feats,
        "original_img": orig_path,
        "skeleton_img": skel_path
    })

@app.get("/webcam")
def webcam():
    return StreamingResponse(
        webcam_generator(model, scaler, le),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.post("/predict_video")
async def predict_video(req: Request, video: UploadFile = File(...), num_frames: int = Form(5)): # Default to 5 to be safe
    video_path = f"uploads/{video.filename}"
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(1, total_frames // num_frames)
    
    frame_results = []
    gemini_payload = [
    "Analyze this baseball swing sequence. Be brief and professional.",
    "1. Identify the single biggest posture flaw across these frames.",
    "2. Provide 3 specific, bulleted 'Pro Tips' to fix it.",
    "3. Focus ONLY on biomechanics (elbows, hips, torso). Use Markdown for bolding.",
    ]
    
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret: break

        skeleton_frame, feats = process_frame_raw(frame)
        score = 0.0
        if feats:
            X_df = pd.DataFrame([[feats[f] for f in FEATURE_ORDER]], columns=FEATURE_ORDER)
            score = float(np.max(model.predict_proba(scaler.transform(X_df))))

        # Save frame for UI
        frame_filename = f"frame_{i}_{video.filename}.jpg"
        frame_path_local = os.path.join("uploads", frame_filename)
        cv2.imwrite(frame_path_local, skeleton_frame if skeleton_frame is not None else frame)

        # Prepare frame for Gemini
        _, buffer = cv2.imencode('.jpg', frame)
        gemini_payload.append(types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg"))
        
        frame_results.append({
            "img": f"/uploads/{frame_filename}",
            "score": round(score * 100, 2)
        })

    # Single API Call for everything
    gemini_payload.append("Format the response using bold headings and bullet points. Keep it under 150 words.")
    
    try:
        # ONE call for the whole video sequence
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=gemini_payload
        )
        strategy_text = response.text
    except Exception as e:
        strategy_text = "The AI is currently busy (Quota reached). Please wait 60 seconds and try again with 3 frames."

    cap.release()
    return templates.TemplateResponse("video_results.html", {
        "request": req, "frames": frame_results, "strategy": strategy_text
    })