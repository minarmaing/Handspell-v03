import os
import json
import logging
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List

# 1. Initialize Logging Matrix
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ASL_LSTM_Sequence_API")

# 2. Workspace File Targets
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "handspell_lstm_v1.keras")
METADATA_PATH = os.path.join(MODELS_DIR, "handspell_lstm_v1_metadata.json")

# 3. Securely Load Keras Neural Network and Lookup Metadata
if os.path.exists(MODEL_PATH) and os.path.exists(METADATA_PATH):
    try:
        # Load the compiled Keras sequence model weights
        lstm_model = tf.keras.models.load_model(MODEL_PATH)

        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
        CLASSES = metadata.get("classes", [chr(i) for i in range(65, 89)])  # Fallback to A-X
        logger.info("Successfully mounted LSTM neural network layers.")
    except Exception as e:
        logger.error(f"Critical error loading model assets: {str(e)}")
        raise RuntimeError(f"Model initialization failure: {e}")
else:
    logger.warning("Recurrent neural weight files not found. Verify deployment copy directives.")
    CLASSES = [chr(i) for i in range(65, 89)]

# 4. Instantiate FastAPI
app = FastAPI(
    title="ASL Time-Series Sequence Landmark API",
    description="Real-time multi-frame LSTM inference endpoint for processing dynamic gestures.",
    version="3.0.0"
)

# 5. Inject Global CORS Middleware Policies
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Define Pydantic 3D Shape Validation Schemas
class FrameData(BaseModel):
    # Validates that a single frame contains exactly 63 tracking landmarks
    features: List[float] = Field(..., min_items=63, max_items=63)

class VideoSequenceInput(BaseModel):
    # Enforces a sequential matrix consisting of exactly 30 chronological frames
    sequence: List[FrameData] = Field(
        ...,
        min_items=30,
        max_items=30,
        description="A list of 30 frames, where each frame contains 63 normalized (x,y,z) coordinate coordinates."
    )

# 7. Endpoints
@app.get("/", status_code=status.HTTP_200_OK)
def health_check():
    """System heartbeat monitored by GCP Cloud Run."""
    return {
        "status": "healthy",
        "lstm_layers_loaded": 'lstm_model' in globals(),
        "expected_sequence_shape": [30, 63]
    }

@app.post("/predict", status_code=status.HTTP_200_OK)
def predict_sequence(payload: VideoSequenceInput):
    """Processes sequential matrices over time to identify dynamic sign gestures."""
    try:
        # Extract the coordinate floats from the incoming Pydantic nested models object
        raw_sequence = [frame.features for frame in payload.sequence]

        # Convert list to a 3D NumPy array: (30, 63)
        sequence_matrix = np.array(raw_sequence)

        # Add the batch dimension required by Keras: (1, 30, 63)
        input_tensor = np.expand_dims(sequence_matrix, axis=0)

        # Execute forward-propagation inference matrix calculation
        probabilities = lstm_model.predict(input_tensor, verbose=0)

        # Parse outputs
        max_idx = int(np.argmax(probabilities[0]))
        confidence = float(probabilities[0][max_idx])
        predicted_gesture = CLASSES[max_idx]

        logger.info(f"Classified structural movement: '{predicted_gesture}' with {confidence*100:.2f}% confidence.")

        return {
            "prediction": predicted_gesture,
            "confidence_score": round(confidence, 4),
            "engine": "LSTM Recurrent Network v1"
        }

    except Exception as error:
        logger.error(f"Runtime sequence inference failure: {str(error)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure processing sequential coordinates: {str(error)}"
        )
