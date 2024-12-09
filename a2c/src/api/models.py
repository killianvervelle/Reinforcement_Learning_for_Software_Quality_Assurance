from pydantic import BaseModel


class modelRequest(BaseModel):
    epochs: int | None = 50
    learning_rate: float | None = 0.01
    data: list


class modelResponse(BaseModel):
    response: list
