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
from pymongo import ReturnDocument
import json
from langchain_community.llms import Ollama
from langgraph.graph import StateGraph, END
import asyncio
import threading


logger = logging.getLogger("inventory_agent")
logging.basicConfig(level=logging.INFO)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://user:pass1@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8001))

llm = Ollama(model="tinyllama", temperature=0.0)

app = FastAPI(title="Inventory Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

lock = threading.Lock()


class CartItem(BaseModel):
    sku: str
    qty: int = Field(1, gt=0)


class Cart(BaseModel):
    cart_id: str
    items: List[CartItem] = []


class ReservationReq(BaseModel):
    order_id: str
    items: List[CartItem] = []
    atomic_update: bool = False
    delay: float = 0.0
    drop: int = 0


class InventoryAgentState(TypedDict):
    order_id: str
    items: List[Dict[str, int]]
    atomic: bool
    action: str
    result: Optional[Dict[str, Any]]


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


async def validate_stock_tool(state: InventoryAgentState) -> InventoryAgentState:
    if state["action"] == 'ROLLBACK':
        return state

    for item in state["items"]:
        doc = await db.inventory.find_one({"sku": item["sku"]})
        if not doc or doc["stock"] < item["qty"]:
            state["action"] = "OUT_OF_STOCK"
            state["result"] = {
                "order_id": state["order_id"],
                "status": "OUT_OF_STOCK",
                "failed_sku": item["sku"]
            }
            return state

    state["action"] = "RESERVABLE"
    return state


async def apply_reservation_tool(state: InventoryAgentState) -> InventoryAgentState:
    results = []

    if state["atomic"]:
        with lock:
            for item in state["items"]:
                res = await db.inventory.find_one_and_update(
                    {"sku": item["sku"]},
                    {"$inc": {"stock": -item["qty"]}},
                    return_document=ReturnDocument.AFTER
                )
                results.append({"sku": item["sku"], "remaining": res["stock"]})
    else:
        for item in state["items"]:
            doc = await db.inventory.find_one({"sku": item["sku"]})
            new_stock = doc["stock"] - item["qty"]
            await db.inventory.update_one(
                {"sku": item["sku"]},
                {"$set": {"stock": new_stock}}
            )
            results.append({"sku": item["sku"], "remaining": new_stock})

    state["result"] = {
        "order_id": state["order_id"],
        "status": "RESERVED",
        "items": results
    }
    return state


async def rollback_reservation_tool(state: InventoryAgentState) -> InventoryAgentState:
    results = []

    if state["atomic"]:
        with lock:
            for item in state["items"]:
                res = await db.inventory.find_one_and_update(
                    {"sku": item["sku"]},
                    {"$inc": {"stock": item["qty"]}},
                    return_document=ReturnDocument.AFTER
                )
                results.append({"sku": item["sku"], "remaining": res["stock"]})
    else:
        for item in state["items"]:
            doc = await db.inventory.find_one({"sku": item["sku"]})
            new_stock = doc["stock"] + item["qty"]
            await db.inventory.update_one(
                {"sku": item["sku"]},
                {"$set": {"stock": new_stock}}
            )
            results.append({"sku": item["sku"], "remaining": new_stock})

    state["result"] = {
        "order_id": state["order_id"],
        "status": "RESERVED_ROLLBACK",
        "items": results
    }
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


async def reasoning_node(state: InventoryAgentState) -> InventoryAgentState:
    prompt = f"""
    You are an inventory reservation agent.
    
    Task: 
    - If Action input is RESERVABLE, respond:
        {{"decision": "APPLY_RESERVE"}}
    
    - else if Action input is OUT_OF_STOCK, respond:
        {{"decision": "OUT_OF_STOCK"}}
    
    - else if Action input is ROLLBACK, respond:
        {{"decision": "ROLLBACK_RESERVE"}}

    Input:
    Action: {state["action"]}
    
    Return ONLY valid JSON.
    """

    raw = await asyncio.to_thread(llm.invoke, prompt)
    print(f'RAW LLM Response: {raw}')

    decision = parse_json_response(raw).get("decision", "OUT_OF_STOCK")

    state["action"] = decision
    return state


def build_inventory_agent():
    g = StateGraph(InventoryAgentState)

    g.add_node("reason", reasoning_node)
    g.add_node("validate", validate_stock_tool)
    g.add_node("apply", apply_reservation_tool)
    g.add_node("rollback", rollback_reservation_tool)

    g.set_entry_point("validate")

    g.add_edge("validate", "reason")

    g.add_conditional_edges(
        "reason",
        lambda s: s["action"],
        {
            "APPLY_RESERVE": "apply",
            "ROLLBACK_RESERVE": "rollback",
            "OUT_OF_STOCK": END
        }
    )

    g.add_edge("apply", END)
    g.add_edge("rollback", END)

    return g.compile()


inventory_graph = build_inventory_agent()


@app.post("/reset_stocks")
def reset_stocks(request: dict):
    """

    :param request:
        {
          "items": [
            {
              "sku": "4cc0770f-91bc-4c0d-a26f-7b872f02ca94",
              "stock": 10
            }
          ]
        }
    :return:
    """
    db.inventory.delete_many({})
    items: List[CartItem] = request["items"]
    for item in items:
        db.inventory.insert_one({"sku": item['sku'], "stock": item['stock']})


@app.post("/reserve")
async def reserve_stock(req: ReservationReq):
    if not req.items:
        raise HTTPException(status_code=400, detail="empty_cart_items")

    state = {
        "order_id": req.order_id,
        "items": [it.dict() for it in req.items],
        "atomic": req.atomic_update,
        "action": None,
        "result": None
    }

    out = await inventory_graph.ainvoke(state)
    return out.get("result")


@app.post("/reserve-rollback")
async def rollback_stock(req: ReservationReq):
    state = {
        "order_id": req.order_id,
        "items": [it.dict() for it in req.items],
        "atomic": req.atomic_update,
        "action": 'ROLLBACK',
        "result": None
    }
    out = await inventory_graph.ainvoke(state)
    return out.get("result")
