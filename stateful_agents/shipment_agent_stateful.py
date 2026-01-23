import os
import logging
import time
import uuid
import datetime
import httpx

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional, Literal
from motor.motor_asyncio import AsyncIOMotorClient
from httpx import AsyncClient
import json
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import asyncio

################################
# load_memory → LLM plan shipment params → carrier_booking_tool(params) → LLM verify
################################

logger = logging.getLogger("shipment_agent")
logging.basicConfig(
    filename='logs/shipment_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8006))

llm = ChatOllama(model="qwen2", temperature=0.0, reasoning=False)

app = FastAPI(title="Shipment Booking Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

class UserMemory(BaseModel):
    user_id: str
    summary: str
    updated_at: datetime.datetime

class ShipmentRequest(BaseModel):
    order_id: str
    address: str


class ShipmentResponse(BaseModel):
    shipment_id: str
    tracking_id: str

class ShipmentPreferences(BaseModel):
    speed: Literal["fastest", "standard", "cheapest"]
    eco_friendly: bool
    avoid_weekend_delivery: bool
    preferred_carrier: Optional[str]

class ShipmentState(TypedDict):
    request: Dict[str, Any]
    user_id: Optional[str]
    memory_summary: Optional[str]
    shipment_prefs: Optional[Dict[str, Any]]
    carrier_result: Dict[str, Any]
    result: Dict[str, Any]


async def load_memory_node(state: ShipmentState) -> ShipmentState:
    user_id = state.get("user_id")
    if not user_id:
        state["memory_summary"] = None
        return state

    doc = await db.user_memory.find_one({"user_id": user_id})
    state["memory_summary"] = doc["summary"] if doc else None
    return state

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

    prefs = state.get("shipment_prefs", {})
    logger.info(f"Booking shipment with prefs: {prefs}")

    # Simulate carrier API latency
    time.sleep(0.2)

    tracking_id = str(uuid.uuid4())
    state["carrier_result"] = {
        "tracking_id": tracking_id,
        "carrier": prefs.get("preferred_carrier", "MockCarrier"),
        "speed": prefs.get("speed", "standard")
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


async def plan_shipment_params(state: ShipmentState) -> ShipmentState:
    memory_block = (
        f"User shipment preference summary:\n{state['memory_summary']}\n\n"
        if state.get("memory_summary")
        else "No user preferences available.\n\n"
    )

    prompt = f"""
    You are a shipment planning agent.

    {memory_block}

    Task:
    - Infer shipment preferences from summarized memory if present
    - If memory is missing, try to infer preferences from user input, otherwise choose safe defaults
    - Return ONLY valid JSON matching the schema

    Defaults:
    - speed = "standard"
    - eco_friendly = false
    - avoid_weekend_delivery = false
    - preferred_carrier = null

    Schema:
    {{
      "speed": "fastest" | "standard" | "cheapest",
      "eco_friendly": bool,
      "avoid_weekend_delivery": bool,
      "preferred_carrier": string | null
    }}
    """

    response = await asyncio.to_thread(llm.invoke, prompt)
    raw = response.text()

    prefs = parse_json_response(raw)
    ShipmentPreferences(**prefs)  # validation

    state["shipment_prefs"] = prefs
    return state


async def shipment_reasoning(state: ShipmentState) -> ShipmentState:
    prompt = f"""
    Verify shipment booking result.

    Rules:
    - tracking_id must exist
    - Generate shipment_id as UUID
    - success = true only if tracking_id exists

    Input:
    CARRIER_RESULT = {json.dumps(state["carrier_result"])}

    Schema:
    {{
      "shipment_id": string,
      "success": bool
    }}
    """

    response = await asyncio.to_thread(llm.invoke, prompt)
    parsed = parse_json_response(response.text())

    state["result"] = parsed
    state["result"]["tracking_id"] = state["carrier_result"]["tracking_id"]
    return state



def build_shipment_graph():
    graph = StateGraph(ShipmentState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("plan_params", plan_shipment_params)
    graph.add_node("carrier_call", carrier_booking_tool)
    graph.add_node("verify", shipment_reasoning)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "plan_params")
    graph.add_edge("plan_params", "carrier_call")
    graph.add_edge("carrier_call", "verify")
    graph.add_edge("verify", END)

    return graph.compile()


shipment_graph = build_shipment_graph()


@app.post("/book", response_model=ShipmentResponse)
async def book_shipment(req: ShipmentRequest, user_id: Optional[str] = Query(None),):
    try:
        state = {
            "request": req.dict(),
            "user_id": user_id,
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
async def clear_bookings():
    await db.shipments.delete_many({})
