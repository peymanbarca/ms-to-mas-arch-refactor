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


logger = logging.getLogger("pricing")
logging.basicConfig(level=logging.INFO)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://user:pass1@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8003))

llm = Ollama(model="qwen2", temperature=0.0)

app = FastAPI(title="Shopping Cart Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

class CartItem(BaseModel):
    sku: str
    qty: int = Field(1, gt=0)

class Cart(BaseModel):
    cart_id: str
    items: List[CartItem] = []


class CartAgentState(TypedDict):
    cart_id: str
    action: str
    item: Dict[str, Any] | None
    cart: Dict[str, Any]
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


# Tool: Fetch cart
async def fetch_cart_tool(state: CartAgentState) -> CartAgentState:
    doc = await db.carts.find_one({"cart_id": state["cart_id"]})
    if not doc:
        raise ValueError("cart not found")

    state["cart"] = {
        "cart_id": doc["cart_id"],
        "items": doc.get("items", [])
    }
    return state


# Tool: Persist Cart Tool
async def persist_cart_tool(state: CartAgentState) -> CartAgentState:
    await db.carts.update_one(
        {"cart_id": state["cart_id"]},
        {"$set": {"items": state["cart"]["items"]}},
        upsert=True
    )
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


async def cart_reasoning_node(state: CartAgentState) -> CartAgentState:
    prompt = f"""
    You are a shopping cart management agent.
    
    You must update the cart state based on the action.
    
    Allowed actions:
    - ADD_ITEM
    - REMOVE_ITEM
    - VIEW
    
    Rules:
    - Cart items are identified by SKU
    - Quantity must always be >= 1
    - Removing an item deletes it entirely
    - Output ONLY valid JSON
    
    Schema:
    {{
      "cart_id": string,
      "items": [
        {{ "sku": string, "qty": number }}
      ]
    }}
    
    Input:
    ACTION = {state["action"]}
    ITEM = {json.dumps(state["item"])}
    CURRENT_CART = {json.dumps(state["cart"])}
    """

    raw = await asyncio.to_thread(llm.invoke, prompt)

    try:
        updated = parse_json_response(raw)
    except Exception as e:
        raise ValueError(f"Invalid JSON from cart agent: {raw}") from e

    state["result"] = updated
    state["cart"]["items"] = updated["items"]
    return state


def build_cart_agent():
    graph = StateGraph(CartAgentState)

    graph.add_node("fetch_cart", fetch_cart_tool)
    graph.add_node("reason_cart", cart_reasoning_node)
    graph.add_node("persist_cart", persist_cart_tool)

    graph.set_entry_point("fetch_cart")
    graph.add_edge("fetch_cart", "reason_cart")
    graph.add_edge("reason_cart", "persist_cart")
    graph.add_edge("persist_cart", END)

    return graph.compile()


cart_graph = build_cart_agent()


@app.get("/cart/{cart_id}", response_model=Cart)
async def get_cart(cart_id: str):
    try:
        state = {
            "cart_id": cart_id,
            "action": "VIEW",
            "item": None,
            "cart": {},
            "result": {}
        }
        out = await cart_graph.ainvoke(state)
        return Cart(**out["cart"])
    except Exception:
        raise HTTPException(status_code=404, detail="cart not found")


@app.post("/cart/{cart_id}/items", response_model=Cart)
async def add_item(cart_id: str, item: CartItem):
    try:
        state = {
            "cart_id": cart_id,
            "action": "ADD_ITEM",
            "item": item.dict(),
            "cart": {},
            "result": {}
        }
        out = await cart_graph.ainvoke(state)
        return Cart(**out["cart"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/cart/{cart_id}/items/{sku}", response_model=Cart)
async def remove_item(cart_id: str, sku: str):
    try:
        state = {
            "cart_id": cart_id,
            "action": "REMOVE_ITEM",
            "item": {"sku": sku},
            "cart": {},
            "result": {}
        }
        out = await cart_graph.ainvoke(state)
        return Cart(**out["cart"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


