from typing import Union
from fastapi import FastAPI, UploadFile, HTTPException
from .ai.recognizer import Recognizer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_FILE_UPLOAD_FORMAT = ["image/jpeg"]
recognizer = Recognizer()


@app.post("/recognize")
def recognize_handler(file: UploadFile):
    if file.content_type not in IMAGE_FILE_UPLOAD_FORMAT:
        raise HTTPException(415, detail='Image must be of type "image/jpeg"')
    return recognizer.recognize(file.file)


@app.post("/enroll")
def enroll_handler():
    raise HTTPException(501, detail="This feature has not been implemented")
