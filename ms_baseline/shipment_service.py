# shipment.py
import os
import logging
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
http_client: httpx.AsyncClient = None

class ShipmentRequest(BaseModel):
    order_id: str
    address: Dict
    items: List[Dict]

class ShipmentResponse(BaseModel):
    shipment_id: str
    carrier: str
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
    payload = {"order_id": req.order_id, "address": req.address, "items": req.items}
    try:
        r = await http_client.post(CARRIER_API, json=payload)
        r.raise_for_status()
        jr = r.json()
        shipment_id = jr.get("shipment_id", str(uuid.uuid4()))
        doc = {
            "shipment_id": shipment_id,
            "order_id": req.order_id,
            "carrier": jr.get("carrier", "Unknown"),
            "tracking_id": jr.get("tracking_id", ""),
            "raw_response": jr
        }
        await db.shipments.insert_one(doc)
        return ShipmentResponse(shipment_id=shipment_id, carrier=doc["carrier"], tracking_id=doc["tracking_id"])
    except httpx.RequestError:
        logger.exception("carrier booking failed")
        raise HTTPException(status_code=502, detail="Carrier unavailable")
