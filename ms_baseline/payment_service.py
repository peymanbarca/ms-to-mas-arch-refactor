from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
from pymongo import MongoClient
import random
import time
import os

app = FastAPI(title="Mock Payment Service")
PAYMENT_COLL = MongoClient("mongodb://user:pass1@localhost:27017/")["ms_baseline"]["payments"]
PORT = int(os.getenv("PORT", 8007))


# -----------------------------
# Pydantic Schemas
# -----------------------------
class PaymentRequest(BaseModel):
    order_id: str = Field(..., example="d02fdb40-c0df-4f44-8247-cedbce182b77")
    final_price: float


class PaymentResponse(BaseModel):
    order_id: str
    status: Literal["SUCCESS", "FAILED"]


@app.post("/clear_payments")
def clear_payments():
    PAYMENT_COLL.delete_many({})


@app.post("/pay-order", response_model=PaymentResponse, summary="Process payment for an order")
def process_payment(request: PaymentRequest):
    """
    Simulate a payment process by calling an external PSP.
    - Randomly determines success (75% success rate by default)
    """
    time.sleep(0.3)

    # success = random.choices([True, False], weights=[3, 1])[0]  # 75% success
    # status = "SUCCESS" if success else "FAILED"
    status = "SUCCESS"

    PAYMENT_COLL.insert_one({'order_id': request.order_id, 'final_price': request.final_price, 'status': status})
    return PaymentResponse(order_id=request.order_id, status=status)
