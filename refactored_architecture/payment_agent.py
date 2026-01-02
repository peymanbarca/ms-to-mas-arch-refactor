import os
import logging
import time
import uuid
import datetime
import httpx

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional, Literal
from motor.motor_asyncio import AsyncIOMotorClient
from httpx import AsyncClient
import json
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
import asyncio


logger = logging.getLogger("pricing")
logging.basicConfig(level=logging.INFO)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://user:pass1@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8007))

llm = Ollama(model="qwen2", temperature=0.0)

app = FastAPI(title="Shipment Booking Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

class PaymentRequest(BaseModel):
    order_id: str
    final_price: float


class PaymentResponse(BaseModel):
    order_id: str
    status: Literal["SUCCESS", "FAILED"]


class PaymentAgentState(TypedDict):
    order_id: str
    psp_tracking_id: Optional[str]
    final_price: float
    decision: Dict[str, Any]


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


# Tool: call external PSP
async def call_external_psp_tool(state: PaymentAgentState) -> PaymentAgentState:
    # Simulate carrier API latency
    time.sleep(0.3)

    psp_tracking_id = str(uuid.uuid4())
    state["psp_tracking_id"] = psp_tracking_id
    return state


# Tool: Persist Payment Result (Deterministic)
async def persist_payment_tool(state: PaymentAgentState) -> PaymentAgentState:
    doc = {
        "order_id": state["order_id"],
        "final_price": state["final_price"],
        "status": state["decision"]["status"],
        "psp_tracking_id": state["psp_tracking_id"],
        "created_at": datetime.datetime.now()
    }
    await db.payments.insert_one(doc)
    return state


def parse_json_response(text: str):
    import re
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        t = text.strip().lower()
        if t in ("reserved", "yes"):
            return {"status": "reserved"}
        if t in ("out_of_stock", "no"):
            return {"status": "out_of_stock"}
        return None
    except Exception as e:
        logging.error(f"parse error: {e} -- {text}")
        return None


async def payment_reasoning_node(state: PaymentAgentState) -> PaymentAgentState:
    prompt = f"""
    You are a payment authorization agent.
    
    You must decide whether a payment succeeds or fails.
    
    Rules:
    - Output MUST be valid JSON only
    - Status must be either "SUCCESS" or "FAILED"
    - if PSP_TRACKING_ID input is not null the Status should be SUCCESS, otherwise it should be FAILED
    
    Schema:
    {{
      "status": "SUCCESS" | "FAILED",
      "reason": string
    }}
    
    Input:
    PSP_TRACKING_ID = {state["psp_tracking_id"]}
    ORDER_ID = {state["order_id"]}
    FINAL_PRICE = {state["final_price"]}
    """

    raw = await asyncio.to_thread(llm.invoke, prompt)

    try:
        decision = parse_json_response(raw)
        assert decision["status"] in ["SUCCESS", "FAILED"]
    except Exception as e:
        raise ValueError(f"Invalid payment decision: {raw}") from e

    state["decision"] = decision
    return state


def build_payment_agent():
    graph = StateGraph(PaymentAgentState)

    graph.add_node("psp_call", call_external_psp_tool)
    graph.add_node("decide_payment", payment_reasoning_node)
    graph.add_node("persist_payment", persist_payment_tool)

    graph.set_entry_point("psp_call")
    graph.add_edge("psp_call", "decide_payment")
    graph.add_edge("decide_payment", "persist_payment")
    graph.add_edge("persist_payment", END)

    return graph.compile()


payment_graph = build_payment_agent()


@app.post("/pay-order", response_model=PaymentResponse, summary="Process payment for an order")
async def process_payment(req: PaymentRequest):
    try:
        state = {
            "order_id": req.order_id,
            "final_price": req.final_price,
            "decision": {}
        }
        out = await payment_graph.ainvoke(state)
        return PaymentResponse(
            order_id=req.order_id,
            status=out["decision"]["status"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/clear_payments")
def clear_payments():
    db.payments.delete_many({})
