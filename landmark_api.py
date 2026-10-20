from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
import json

import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from xgboost import XGBClassifier


MODEL_DIR = Path(__file__).resolve().parent / "models"
MAX_FILE_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    metadata = json.loads(
        (MODEL_DIR / "handspell_xgboost_v1_metadata.json").read_text(
            encoding="utf-8"
        )
    )

    classifier = XGBClassifier()
    classifier.load_model(
        str(MODEL_DIR / "handspell_xgboost_v1.json")
    )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_DIR / "hand_landmarker.task")
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:
        app.state.classifier = classifier
        app.state.landmarker = landmarker
        app.state.letters = metadata["letters"]

        # Serialize access to the shared inference objects.
        app.state.lock = Lock()

        yield


app = FastAPI(
    title="HandSpell — MediaPipe + XGBoost",
    lifespan=lifespan,
)


def read_rgb(contents):
    # Same image preparation as the training notebook.
    with Image.open(BytesIO(contents)) as original:
        oriented = ImageOps.exif_transpose(original).convert("RGBA")
        background = Image.new(
            "RGBA", oriented.size, (255, 255, 255, 255)
        )
        image = Image.alpha_composite(
            background, oriented
        ).convert("RGB")

        image.thumbnail(
            (1024, 1024),
            Image.Resampling.LANCZOS,
        )

        return np.array(image)


def get_features(rgb, landmarker):
    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb),
    )

    result = landmarker.detect(image)

    if not result.hand_landmarks:
        return None

    points = np.array(
        [
            [point.x, point.y, point.z]
            for point in result.hand_landmarks[0]
        ],
        dtype=np.float32,
    )

    height, width = rgb.shape[:2]
    points *= np.array(
        [width, height, width],
        dtype=np.float32,
    )

    relative = points - points[0]
    scale = np.linalg.norm(relative[9])

    if scale < 1e-6:
        return None

    return (relative / scale).reshape(1, 63)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "MediaPipe + XGBoost",
        "classes": len(app.state.letters),
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read(MAX_FILE_BYTES + 1)
    finally:
        file.file.close()

    if not contents:
        raise HTTPException(400, "The uploaded file is empty.")

    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(413, "Maximum image size is 10 MB.")

    try:
        rgb = read_rgb(contents)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise HTTPException(400, "Upload a valid JPG or PNG image.")

    with app.state.lock:
        features = get_features(rgb, app.state.landmarker)

        if features is None:
            raise HTTPException(
                422,
                "No usable hand detected. Try a clearer image.",
            )

        scores = app.state.classifier.predict_proba(features)[0]
        best_index = int(np.argmax(scores))
        class_id = int(app.state.classifier.classes_[best_index])

    return {
        "prediction": app.state.letters[class_id],
        "confidence": float(scores[best_index]),
    }
