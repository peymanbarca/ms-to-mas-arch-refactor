# shipment.py
import datetime
import os
import logging
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from motor.motor_asyncio import AsyncIOMotorClient
import httpx
import uuid

logger = logging.getLogger("shipment")
logging.basicConfig(level=logging.INFO)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://user:pass1@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
CARRIER_API = os.getenv("CARRIER_API", "http://localhost:9020/carrier/book")
PORT = int(os.getenv("PORT", 8006))

app = FastAPI(title="Shipment Service")

db_client = None
db = None
http_client: httpx.AsyncClient = httpx.AsyncClient(timeout=10)


class ShipmentRequest(BaseModel):
    order_id: str
    address: str


class ShipmentResponse(BaseModel):
    shipment_id: str
    tracking_id: str


@app.on_event("startup")
async def startup():
    global db_client, db, http_client
    db_client = AsyncIOMotorClient(MONGO_URI)
    db = db_client[MONGO_DB]
    await db.shipments.create_index("shipment_id", unique=True)
    http_client = httpx.AsyncClient(timeout=10.0)
    logger.info("Shipment connected to mongo")


@app.on_event("shutdown")
async def shutdown():
    global db_client, http_client
    if http_client:
        await http_client.aclose()
    if db_client:
        db_client.close()


@app.post("/book", response_model=ShipmentResponse)
async def book_shipment(req: ShipmentRequest):
    try:
        # simulate calling an external service
        # r = await http_client.post(CARRIER_API, json=payload)
        # r.raise_for_status()
        # jr = r.json()
        time.sleep(0.2)
        tracking_id = str(uuid.uuid4())

        shipment_id = str(uuid.uuid4())
        doc = {
            "shipment_id": shipment_id,
            "order_id": req.order_id,
            "address": req.address,
            "created_at": datetime.datetime.now(),
            "tracking_id": tracking_id
        }
        await db.shipments.insert_one(doc)
        return ShipmentResponse(shipment_id=shipment_id, tracking_id=doc["tracking_id"])
    except httpx.RequestError:
        logger.exception("carrier booking failed")
        raise HTTPException(status_code=502, detail="Carrier unavailable")
