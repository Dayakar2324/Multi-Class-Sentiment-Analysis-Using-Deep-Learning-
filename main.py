
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# import re
# from pydantic import BaseModel, Field
# from keras.models import load_model
# import pickle
# from contextlib import asynccontextmanager
# from fastapi.responses import FileResponses

# model_path = "Artifacts/improved_GRU_model.keras"
# tokenizer_path = "Artifacts/tokenizer.pkl"
# max_length = 100

# emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# emotion_emojis = {
#     "sadness": "😢",
#     "joy": "😊",
#     "love": "❤️",
#     "anger": "😡",
#     "fear": "😨",
#     "surprise": "😲"
# }

# # preprocess the text through regex

# def preprocess_text(text: str) -> str:
#     text = text.lower()
#     text = re.sub("'", "", text)
#     text = re.sub(r"[^a-z0-9\s]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text


# class TextInput(BaseModel):
#     text: str = Field(
#         ...,
#         min_length=1,
#         max_length=2000,
#         json_schema_extra={"example": "i am good"},
#         description="Enter the text for which you want to analyze the sentiment"
#     )


# class Prediction_Response(BaseModel):
#     text: str
#     predicted_emotion: str
#     confidence: float
#     all_probabilities: dict[str, float]


# class Health_response(BaseModel):
#     status: str
#     Model_loaded: bool


# dl_model = {}


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     print("loading the model and tokenizer")

#     dl_model["BIGRU_model"] = load_model(model_path)

#     with open(tokenizer_path, "rb") as file:
#         dl_model["tokenizer"] = pickle.load(file)

#     print("Model is loaded successfully")

#     try:
#         yield
#     finally:
#         dl_model.clear()


# app = FastAPI(lifespan=lifespan)





# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
#     allow_credentials=True
# )


# app.mount('/static', StaticFiles(directory="static"), name="static")


# @app.get('/',include_in_schema = False)

# def server_ui():
#     return FileResponse('static/index.html')

# @app.get('/health',response_model = HealthResponse)
# def health_check():
#     return HealthResponse(status = "server is running",model_loaded = bool(dl_model))

# @app.post('/prediction',response_model = PredictionResponse)
# def Predict_emotion(text_input = TextInput):
#     BIGRU_Model = dl_model.get("BiGRU")
#     tokenizer_Model = dl_model.get("Tokenizer")

#     if BiGRU_model is NONE or tokenizer_Model is None:
#         raise HTTPException(status_code = 503,detail = "Model is not loaded properly , Please load it again")

#     cleaned_text = preprocess_text(text_Input.text)
#     tokenized_text = tokenizer_model.texts_to_sequences(cleaned_text)
#     padded_sequence = pad_sequence(
#         tokenized_text,
#         maxlen = max_sequence_length,
#         padding = 'post',
#         truncating = 'post'

#     )

#     probabilites = BiGRU_model.predict(padded_sequence)[0]

#     top_emotion_index = int(np.argmax(probabilites))
#     label: float(prob) for prob,label in zip(emotion_labels ,probabilites)

#     return PredictionResponse{
#         text = text_input.text,
#         predicted_emotion = float(emotion_labels[top_emotion_index])
#         all_probabilities = probabilities
#     }



import re
import pickle
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from pydantic import BaseModel, Field

from keras.models import load_model
from keras.utils import pad_sequences


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "Artifacts" / "improved_GRU_model.keras"
TOKENIZER_PATH = BASE_DIR / "Artifacts" / "tokenizer.pkl"
STATIC_DIR = BASE_DIR / "static"

MAX_LENGTH = 100

EMOTION_LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise"
]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😡",
    "fear": "😨",
    "surprise": "😲"
}


# --------------------------------------------------
# Model storage
# --------------------------------------------------

dl_model = {
    "model": None,
    "tokenizer": None
}


# --------------------------------------------------
# Text preprocessing
# --------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Clean input text before tokenization.
    """

    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# Request and response schemas
# --------------------------------------------------

class TextInput(BaseModel):

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Enter the text for emotion prediction",
        json_schema_extra={"example": "I am feeling very happy today"}
    )


class PredictionResponse(BaseModel):

    text: str
    predicted_emotion: str
    confidence: float
    emoji: str
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):

    status: str
    model_loaded: bool


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Loading GRU model and tokenizer...")

    try:

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        if not TOKENIZER_PATH.exists():
            raise FileNotFoundError(
                f"Tokenizer file not found: {TOKENIZER_PATH}"
            )

        # Load trained model
        dl_model["model"] = load_model(str(MODEL_PATH))

        # Load tokenizer
        with open(TOKENIZER_PATH, "rb") as file:
            dl_model["tokenizer"] = pickle.load(file)

        print("GRU model loaded successfully.")
        print("Tokenizer loaded successfully.")

    except Exception as error:

        dl_model.clear()
        print(f"Error loading model: {error}")
        raise

    yield

    # Application shutdown
    dl_model.clear()

    print("Model resources released.")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Emotion Detection API",
    description="Emotion prediction using a trained BiGRU model",
    version="1.0.0",
    lifespan=lifespan
)


# --------------------------------------------------
# CORS configuration
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False
)


# --------------------------------------------------
# Static files
# --------------------------------------------------

if STATIC_DIR.exists():

    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )


# --------------------------------------------------
# Home route
# --------------------------------------------------

@app.get("/", include_in_schema=False)
def server_ui():

    index_file = STATIC_DIR / "index.html"

    if not index_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Frontend index.html not found"
        )

    return FileResponse(str(index_file))


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse
)
def health_check():

    model_loaded = (
        dl_model.get("model") is not None
        and dl_model.get("tokenizer") is not None
    )

    return HealthResponse(
        status="Server is running",
        model_loaded=model_loaded
    )


# --------------------------------------------------
# Emotion prediction
# --------------------------------------------------

@app.post(
    "/prediction",
    response_model=PredictionResponse
)
def predict_emotion(text_input: TextInput):

    # Check model availability
    model = dl_model.get("model")
    tokenizer = dl_model.get("tokenizer")

    if model is None or tokenizer is None:

        raise HTTPException(
            status_code=503,
            detail="Model or tokenizer is not loaded properly."
        )

    try:

        # Step 1: Preprocess input
        cleaned_text = preprocess_text(text_input.text)

        if not cleaned_text:
            raise HTTPException(
                status_code=422,
                detail="Input contains no valid text after preprocessing."
            )

        # Step 2: Tokenize input
        tokenized_text = tokenizer.texts_to_sequences(
            [cleaned_text]
        )

        # Step 3: Pad token sequence
        padded_sequence = pad_sequences(
            tokenized_text,
            maxlen=MAX_LENGTH,
            padding="post",
            truncating="post"
        )

        # Step 4: Model prediction
        probabilities = model.predict(
            padded_sequence,
            verbose=0
        )[0]

        # Step 5: Validate output
        if len(probabilities) != len(EMOTION_LABELS):
            raise ValueError(
                "Model output count does not match emotion labels."
            )

        # Step 6: Get highest probability
        top_emotion_index = int(np.argmax(probabilities))

        predicted_emotion = EMOTION_LABELS[top_emotion_index]

        confidence = float(probabilities[top_emotion_index])

        # Step 7: Map probabilities to emotion labels
        all_probabilities = {
            label: float(probability)
            for label, probability in zip(
                EMOTION_LABELS,
                probabilities
            )
        }

        # Step 8: Return response
        return PredictionResponse(
            text=text_input.text,
            predicted_emotion=predicted_emotion,
            confidence=confidence,
            emoji=EMOTION_EMOJIS[predicted_emotion],
            all_probabilities=all_probabilities
        )

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )







