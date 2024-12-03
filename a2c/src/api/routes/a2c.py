from fastapi import APIRouter

from a2c.src.models import modelRequest, modelResponse


router = APIRouter(prefix="/model", tags=["model"])

model = "put logic here"

@router.post("/train")
def train_model(request=modelRequest):
       return

@router.post("/predict")
def predict(request=modelRequest):
      return
