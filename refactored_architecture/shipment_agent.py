import os
import logging
import time
import uuid
import datetime
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from httpx import AsyncClient
import json
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
import asyncio


logger = logging.getLogger("shipment_agent")
logging.basicConfig(
    filename='logs/shipment_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://user:pass1@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8006))

llm = Ollama(model="qwen2", temperature=0.0)

app = FastAPI(title="Shipment Booking Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

class ShipmentRequest(BaseModel):
    order_id: str
    address: str


class ShipmentResponse(BaseModel):
    shipment_id: str
    tracking_id: str


class ShipmentState(TypedDict):
    request: Dict[str, Any]
    carrier_result: Dict[str, Any]
    result: Dict[str, Any]

@app.on_event("startup")
async def startup():
    global db_client, db
    db_client = AsyncIOMotorClient(MONGO_URI)
    db = db_client[MONGO_DB]
    logger.info("Connected to MongoDB at %s db=%s", MONGO_URI, MONGO_DB)


@app.on_event("shutdown")
async def shutdown():
    global db_client
    if db_client:
        db_client.close()
        logger.info("MongoDB connection closed")


# Tool: call external carrier
async def carrier_booking_tool(state: ShipmentState) -> ShipmentState:
    logger.info(f'Calling carrier_booking_tool ... \n Current State is {state}')
    print(f'Calling carrier_booking_tool ... \n Current State is {state}')

    # Simulate carrier API latency
    time.sleep(0.2)

    tracking_id = str(uuid.uuid4())
    state["carrier_result"] = {
        "tracking_id": tracking_id,
        "carrier": "MockCarrier"
    }
    logger.info(f'Response state of carrier_booking_tool ==> {state}, \n-------------------------------------')
    print(f'Response state of carrier_booking_tool ==> {state}, \n-------------------------------------')
    return state


def parse_json_response(text: str):
    import re
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        return None
    except Exception as e:
        logging.error(f"parse error: {e} -- {text}")
        return None


async def shipment_reasoning(state: ShipmentState) -> ShipmentState:
    prompt = f"""
    You are a shipment booking agent in a retail supply chain.

    Your task:
    - Confirm existence of tracking_id in CARRIER_RESULT input
    - Generate shipment_id as uuid
    - Return ONLY response only in JSON in the schema below

    Schema:
    {{
      "shipment_id": string,
      "tracking_id": string,
      "success" bool
    }}

    Input:
    REQUEST = {json.dumps(state["request"])}
    CARRIER_RESULT = {json.dumps(state["carrier_result"])}

    Rules:
    - tracking_id must come from CARRIER_RESULT input
    - shipment_id must be newly generated as UUID
    - if both shipment_id and tracking_id exist, success in response should be true, otherwise it should be false.
    """

    logger.info(f'LLM Call Prompt: {prompt}')
    raw = await asyncio.to_thread(llm.invoke, prompt)
    logger.info(f'LLM Raw response: {raw}')
    print(f'LLM Raw response: {raw}')

    try:
        parsed = parse_json_response(raw)
    except Exception as e:
        logger.info(f'Invalid JSON from shipment agent: {raw}, {e}')
        print(f'Invalid JSON from shipment agent: {raw}, {e}')
        raise ValueError(f"Invalid JSON from shipment agent: {raw}") from e

    state["result"] = parsed
    return state


def build_shipment_graph():
    graph = StateGraph(ShipmentState)

    graph.add_node("carrier_call", carrier_booking_tool)
    graph.add_node("reason_shipment", shipment_reasoning)

    graph.set_entry_point("carrier_call")
    graph.add_edge("carrier_call", "reason_shipment")
    graph.add_edge("reason_shipment", END)

    return graph.compile()


shipment_graph = build_shipment_graph()


@app.post("/book", response_model=ShipmentResponse)
async def book_shipment(req: ShipmentRequest):
    try:
        state = {
            "request": req.dict(),
            "carrier_result": {},
            "result": {}
        }
        logger.info(f'Request for book_shipment, req = {req}, state={state}')
        print(f'Request for book_shipment, req = {req}, state={state}')

        out = await shipment_graph.ainvoke(state)
        logger.info(f'Request for process_payment processed successfully, req = {req}, result={out.get("result")}')
        print(f'Request for process_payment processed successfully, req = {req}, result={out.get("result")}')

        success = out["result"]["success"]
        if success is None or success is not True:
            raise HTTPException(status_code=500, detail='Carrier unavailable')

        shipment_id = out["result"]["tracking_id"]
        tracking_id = out["result"]["shipment_id"]

        doc = {
            "shipment_id": shipment_id,
            "order_id": req.order_id,
            "address": req.address,
            "tracking_id": tracking_id,
            "created_at": datetime.datetime.utcnow()
        }

        await db.shipments.insert_one(doc)

        return ShipmentResponse(**out["result"])

    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/clear_bookings")
def clear_bookings():
    db.shipments.delete_many({})
