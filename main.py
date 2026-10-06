from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from face_utils import enroll_face, verify_face

app = FastAPI(title="SmartAttend ML Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EnrollRequest(BaseModel):
    user_id: int
    image_base64: str


class VerifyRequest(BaseModel):
    user_id: int
    image_base64: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/enroll")
def enroll(request: EnrollRequest):
    success = enroll_face(request.user_id, request.image_base64)
    if not success:
        return {"success": False, "message": "No face detected"}
    return {"success": True, "message": "Face enrolled"}


@app.post("/verify")
def verify(request: VerifyRequest):
    match, confidence = verify_face(request.user_id, request.image_base64)
    return {"match": match, "confidence": confidence}
