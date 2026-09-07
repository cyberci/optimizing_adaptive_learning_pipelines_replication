"""FastAPI reproduction of the recommendation endpoint.

Used only by the optional HTTP cross-check of the concurrency measurement. The
runtime variant is selected through the SUGGEST_VARIANT environment variable."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
import engine

VARIANT = os.environ.get("SUGGEST_VARIANT", "V4")
MODEL = os.environ.get("SUGGEST_MODEL", "medusa_B_corrected")
K = 10
ETA = 0.15                        # step size of the profile update rule
EXERCISE = [0, 0.8, 0, 0, 0.8, 0, 0.85, 0, 1.4, 0.75]

app = FastAPI()
eng = engine.Engine(VARIANT, desc={**engine.DESCRIPTOR, "id": MODEL})


class SuggestDTO(BaseModel):
    abilities: List[float]
    prev_params: Dict[str, float]
    result: bool


class SuggestResponseDTO(BaseModel):
    suggested_params: Dict[str, float]
    success_rate: float
    abilities_if_success: List[float]
    abilities_if_failure: List[float]


@app.post("/games/{game_id}/suggest", response_model=SuggestResponseDTO)
def suggest(game_id: str, body: SuggestDTO):
    params, rate, _ = eng.suggest(np.asarray(body.abilities, np.float32),
                                  body.prev_params, body.result)
    up = [a + (0.0 if EXERCISE[i] == 0 else ETA * (1 - rate)) for i, a in enumerate(body.abilities)]
    dn = [a + (0.0 if EXERCISE[i] == 0 else ETA * (0 - rate)) for i, a in enumerate(body.abilities)]
    return SuggestResponseDTO(suggested_params=params, success_rate=rate,
                              abilities_if_success=up, abilities_if_failure=dn)


@app.get("/healthz")
def healthz():
    return {"variant": VARIANT, "model": MODEL}
