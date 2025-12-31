from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import random
import time

app = FastAPI(title="Mock Payment Service")


# -----------------------------
# Pydantic Schemas
# -----------------------------
class PaymentRequest(BaseModel):
    order_id: str = Field(..., example="d02fdb40-c0df-4f44-8247-cedbce182b77")


class PaymentResponse(BaseModel):
    order_id: str
    status: Literal["SUCCESS", "FAILED"]


@app.post("/pay", response_model=PaymentResponse, summary="Process payment for an order")
def process_payment(request: PaymentRequest):
    """
    Simulate a payment process.
    - Randomly determines success (75% success rate by default)
    """
    time.sleep(1)

    success = random.choices([True, False], weights=[3, 1])[0]  # 75% success
    status = "SUCCESS" if success else "FAILED"

    return PaymentResponse(order_id=request.order_id, status=status)
