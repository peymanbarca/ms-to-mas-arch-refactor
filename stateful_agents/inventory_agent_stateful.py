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
from pymongo import ReturnDocument
import json
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
import asyncio
import threading


logger = logging.getLogger("inventory_agent")
logging.basicConfig(
    filename='logs/inventory_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8001))

llm = ChatOllama(model="qwen2", temperature=0.0, reasoning=False)

app = FastAPI(title="Inventory Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

lock = threading.Lock()


class CartItem(BaseModel):
    sku: str
    qty: int = Field(1, gt=0)


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


class LedgerEvent(BaseModel):
    event_id: str
    order_id: str
    sku: str
    qty: int
    event_type: Literal[
        "RESERVE_ATTEMPT",
        "RESERVE_SUCCESS",
        "RESERVE_FAILED",
        "ROLLBACK"
    ]
    stock_before: int
    stock_after: int
    timestamp: datetime.datetime


async def write_ledger_event(
    order_id: str,
    sku: str,
    qty: int,
    event_type: str,
    stock_before: int,
    stock_after: int
):
    doc = {
        "event_id": str(uuid.uuid4()),
        "order_id": order_id,
        "sku": sku,
        "qty": qty,
        "event_type": event_type,
        "stock_before": stock_before,
        "stock_after": stock_after,
        "timestamp": datetime.datetime.utcnow()
    }
    await db.inventory_ledger.insert_one(doc)


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


# Tool: Extended validate_stock_tool with ledger attempt
async def validate_stock_tool(state: InventoryAgentState) -> InventoryAgentState:
    logger.info(f'Calling validate_stock_tool ... \n Current State is {state}')
    print(f'Calling validate_stock_tool ... \n Current State is {state}')

    if state["action"] == 'ROLLBACK':
        return state

    for item in state["items"]:
        doc = await db.inventory.find_one({"sku": item["sku"]})
        stock = doc["stock"] if doc else 0

        await write_ledger_event(
            order_id=state["order_id"],
            sku=item["sku"],
            qty=item["qty"],
            event_type="RESERVE_ATTEMPT",
            stock_before=stock,
            stock_after=stock
        )

        if stock < item["qty"]:
            await write_ledger_event(
                order_id=state["order_id"],
                sku=item["sku"],
                qty=item["qty"],
                event_type="RESERVE_FAILED",
                stock_before=stock,
                stock_after=stock
            )

            state["action"] = "OUT_OF_STOCK"
            state["result"] = {
                "order_id": state["order_id"],
                "status": "OUT_OF_STOCK",
                "failed_sku": item["sku"]
            }
            logger.info(f'Response state of validate_stock_tool tool ==> {state}, \n-------------------------------------')
            print(f'Response state of validate_stock_tool tool ==> {state}, \n-------------------------------------')
            return state

    state["action"] = "RESERVABLE"

    logger.info(f'Response state of validate_stock_tool ==> {state}, \n-------------------------------------')
    print(f'Response state of validate_stock_tool ==> {state}, \n-------------------------------------')
    return state


# Tool: Extended apply with ledger
async def apply_reservation_tool(state: InventoryAgentState) -> InventoryAgentState:
    logger.info(f'Calling apply_reservation_tool ... \n Current State is {state}')
    print(f'Calling apply_reservation_tool ... \n Current State is {state}')
    results = []

    if state["atomic"]:
        with lock:
            for item in state["items"]:
                doc = await db.inventory.find_one({"sku": item["sku"]})
                before = doc["stock"]

                res = await db.inventory.find_one_and_update(
                    {"sku": item["sku"]},
                    {"$inc": {"stock": -item["qty"]}},
                    return_document=ReturnDocument.AFTER
                )

                after = res["stock"]

                await write_ledger_event(
                    state["order_id"],
                    item["sku"],
                    item["qty"],
                    "RESERVE_SUCCESS",
                    before,
                    after
                )

                results.append({"sku": item["sku"], "remaining": after})
    else:
        for item in state["items"]:
            doc = await db.inventory.find_one({"sku": item["sku"]})
            before = doc["stock"]
            new_stock = doc["stock"] - item["qty"]
            await db.inventory.update_one(
                {"sku": item["sku"]},
                {"$set": {"stock": new_stock}}
            )

            await write_ledger_event(
                state["order_id"],
                item["sku"],
                item["qty"],
                "RESERVE_SUCCESS",
                stock_before=before,
                stock_after=new_stock
            )

            results.append({"sku": item["sku"], "remaining": new_stock})

    state["result"] = {
        "order_id": state["order_id"],
        "status": "RESERVED",
        "items": results
    }
    logger.info(f'Response state of apply_reservation_tool ==> {state["result"]}, \n-----------------------------')
    print(f'Response state of apply_reservation_tool ==> {state["result"]}, \n---------------------------------')

    return state


# Tool: Extend rollback with ledger
async def rollback_reservation_tool(state: InventoryAgentState) -> InventoryAgentState:
    logger.info(f'Calling rollback_reservation_tool ... \n Current State is {state}')
    print(f'Calling rollback_reservation_tool ... \n Current State is {state}')
    results = []

    if state["atomic"]:
        with lock:
            for item in state["items"]:
                doc = await db.inventory.find_one({"sku": item["sku"]})
                before = doc["stock"]

                res = await db.inventory.find_one_and_update(
                    {"sku": item["sku"]},
                    {"$inc": {"stock": item["qty"]}},
                    return_document=ReturnDocument.AFTER
                )

                after = res["stock"]

                await write_ledger_event(
                    state["order_id"],
                    item["sku"],
                    item["qty"],
                    "ROLLBACK",
                    before,
                    after
                )

                results.append({"sku": item["sku"], "remaining": res["stock"]})
    else:
        for item in state["items"]:
            doc = await db.inventory.find_one({"sku": item["sku"]})
            before = doc["stock"]
            new_stock = doc["stock"] + item["qty"]
            await db.inventory.update_one(
                {"sku": item["sku"]},
                {"$set": {"stock": new_stock}}
            )

            await write_ledger_event(
                state["order_id"],
                item["sku"],
                item["qty"],
                "ROLLBACK",
                stock_before=before,
                stock_after=new_stock
            )

            results.append({"sku": item["sku"], "remaining": new_stock})

    state["result"] = {
        "order_id": state["order_id"],
        "status": "RESERVED_ROLLBACK",
        "items": results
    }
    logger.info(f'Response state of rollback_reservation_tool ==> {state["result"]}, \n-----------------------------')
    print(f'Response state of rollback_reservation_tool ==> {state["result"]}, \n---------------------------------')
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
    - If Action input is None or null, respond:
        {{"decision": "VALIDATE_STOCK"}}
        
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

    logger.info(f'LLM Call Prompt: {prompt}')
    response = await asyncio.to_thread(llm.invoke, prompt)

    raw_response = response.text()
    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    reasoning_text = response.additional_kwargs.get("reasoning_content", None)
    reasoning_tokens = response.usage_metadata.get("output_token_details", {}).get("reasoning", 0)

    print(f'LLM Reasoning Text: {reasoning_text}')
    logger.info(f'LLM Raw response: {raw_response}')
    print(f'LLM Raw response: {raw_response}')

    logger.info(f'LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
                f' reasoning_tokens: {reasoning_tokens}, total_tokens: {total_tokens}')
    print(f'LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
                f' reasoning_tokens: {reasoning_tokens}, total_tokens: {total_tokens}')

    decision = parse_json_response(raw_response).get("decision", "OUT_OF_STOCK")

    state["action"] = decision
    return state


def build_inventory_agent():
    g = StateGraph(InventoryAgentState)

    g.add_node("reason", reasoning_node)
    g.add_node("validate", validate_stock_tool)
    g.add_node("apply", apply_reservation_tool)
    g.add_node("rollback", rollback_reservation_tool)

    g.set_entry_point("reason")

    g.add_conditional_edges(
        "reason",
        lambda s: s["action"],
        {
            "VALIDATE_STOCK": "validate",
            "APPLY_RESERVE": "apply",
            "ROLLBACK_RESERVE": "rollback",
            "OUT_OF_STOCK": END
        }
    )

    g.add_edge("validate", "reason")
    g.add_edge("apply", END)
    g.add_edge("rollback", END)

    return g.compile()


inventory_graph = build_inventory_agent()


@app.post("/reset_stocks")
async def reset_stocks(request: dict):
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
    await db.inventory.delete_many({})
    items: List[CartItem] = request["items"]
    for item in items:
        await db.inventory.insert_one({"sku": item['sku'], "stock": item['stock']})


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

    logger.info(f'Request for reserve_stock, req = {req}, state={state}')
    print(f'Request for reserve_stock, req = {req}, state={state}')

    out = await inventory_graph.ainvoke(state)

    logger.info(f'Request for reserve_stock processed successfully, req = {req}, result={out.get("result")}')
    print(f'Request for reserve_stock processed successfully, req = {req}, result={out.get("result")}')
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
