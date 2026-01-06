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


logger = logging.getLogger("shopping_cart_agent")
logging.basicConfig(
    filename='logs/shopping_cart_agent.log',
    level=logging.INFO,  # Log all messages with severity DEBUG or higher
    format='%(asctime)s - %(levelname)s - %(message)s'  # Define the message format
)

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
    logger.info(f'Calling fetch_cart_tool ... \n Current State is {state}')
    print(f'Calling fetch_cart_tool ... \n Current State is {state}')

    doc = await db.carts.find_one({"cart_id": state["cart_id"]})
    if not doc:
        logger.exception('Response state of fetch_cart_tool ==> cart not found, \n-------------------------------------')
        print('Response state of fetch_cart_tool ==> cart not found, \n-------------------------------------')
        raise ValueError("cart not found")

    state["cart"] = {
        "cart_id": doc["cart_id"],
        "items": doc.get("items", [])
    }

    logger.info(f'Response state of fetch_cart_tool ==> {state}, \n-------------------------------------')
    print(f'Response state of fetch_cart_tool ==> {state}, \n-------------------------------------')
    return state


# Tool: Persist Cart Tool
async def persist_cart_tool(state: CartAgentState) -> CartAgentState:
    logger.info(f'Calling persist_cart_tool ... \n Current State is {state}')
    print(f'Calling persist_cart_tool ... \n Current State is {state}')

    await db.carts.update_one(
        {"cart_id": state["cart_id"]},
        {"$set": {"items": state["cart"]["items"]}},
        upsert=True
    )
    logger.info('Called successfully of persist_cart_tool')
    print('Called successfully of persist_cart_tool')
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

    logger.info(f'LLM Call Prompt: {prompt}')
    raw = await asyncio.to_thread(llm.invoke, prompt)
    logger.info(f'LLM Raw response: {raw}')
    print(f'LLM Raw response: {raw}')

    try:
        updated = parse_json_response(raw)
    except Exception as e:
        logger.info(f'Invalid JSON from cart agent: {raw}, {e}')
        print(f'Invalid JSON from cart agent: {raw}, {e}')
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
        logger.info(f'Request for get_cart, cart_id = {cart_id}, state={state}')
        print(f'Request for get_cart, cart_id = {cart_id}, state={state}')

        out = await cart_graph.ainvoke(state)
        logger.info(f'Request for get_cart processed successfully, cart_id = {cart_id}, result={out.get("cart")}')
        print(f'Request for get_cart processed successfully, cart_id = {cart_id}, result={out.get("cart")}')
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
        logger.info(f'Request for add_item, cart_id = {cart_id}, item = {item}, state={state}')
        print(f'Request for add_item, cart_id = {cart_id}, item = {item}, state={state}')

        out = await cart_graph.ainvoke(state)
        logger.info(f'Request for add_item processed successfully, cart_id = {cart_id}, result={out.get("cart")}')
        print(f'Request for add_item processed successfully, cart_id = {cart_id}, result={out.get("cart")}')
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


