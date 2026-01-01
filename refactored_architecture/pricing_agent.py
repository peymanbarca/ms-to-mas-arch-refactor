import os
import logging
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

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8002))

llm = Ollama(model="qwen2", temperature=0.0)

app = FastAPI(title="Pricing & Promotion Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

class PriceItem(BaseModel):
    product_id: str
    price: float

class PriceRequestItem(BaseModel):
    product_id: str
    qty: int = Field(1, gt=0)

class PriceRequest(BaseModel):
    items: List[PriceRequestItem]
    promo_codes: Optional[List[str]] = None
    currency: Optional[str] = "USD"

class PriceResponseItem(BaseModel):
    product_id: str
    qty: int
    unit_price: float
    line_total: float
    discounts: float

class PriceResponse(BaseModel):
    items: List[PriceResponseItem]
    subtotal: float
    total_discount: float
    total: float
    currency: str

class PricingState(TypedDict):
    request: Dict[str, Any]
    price_map: Dict[str, float]
    result: Dict[str, Any]

@app.on_event("startup")
async def startup():
    global db_client, db
    db_client = AsyncIOMotorClient(MONGO_URI)
    db = db_client[MONGO_DB]
    # Ensure index
    await db.prices.create_index("product_id", unique=True)
    logger.info("Connected to MongoDB at %s db=%s", MONGO_URI, MONGO_DB)


@app.on_event("shutdown")
async def shutdown():
    global db_client
    if db_client:
        db_client.close()
        logger.info("MongoDB connection closed")

# Tool: Fetch Prices from MongoDB
def make_fetch_prices():
    async def fetch_prices(state: PricingState) -> PricingState:
        items = state["request"]["items"]
        product_ids = [i["product_id"] for i in items]

        docs = await db.prices.find(
            {"product_id": {"$in": product_ids}}
        ).to_list(length=len(product_ids))

        price_map = {d["product_id"]: d["price"] for d in docs}

        missing = set(product_ids) - set(price_map.keys())
        if missing:
            raise ValueError(f"Missing prices for products: {missing}")

        state["price_map"] = price_map
        return state

    return fetch_prices


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


async def pricing_reasoning(state: PricingState) -> PricingState:
    prompt = f"""
You are a pricing agent in a retail supply chain.

You MUST compute prices using the unit prices provided.
Apply promotions if applicable.
Return ONLY valid JSON in the following schema:

{{
  "items": [
    {{
      "product_id": string,
      "qty": number,
      "unit_price": number,
      "line_total": number,
      "discounts": number
    }}
  ],
  "subtotal": number,
  "total_discount": number,
  "total": number,
  "currency": string
}}

Input:
REQUEST = {json.dumps(state["request"])}
PRICE_MAP = {json.dumps(state["price_map"])}

Promotion semantics, Only apply if promo_codes in REQUEST is not empty:
- PROMO10 → 10% off line total
- BUYS2SAVE5 → $5 off if qty >= 2
"""

    # LangChain Ollama is synchronous → offload
    raw = await asyncio.to_thread(llm.invoke, prompt)
    print(f'RAW LLM Response: {raw}')

    try:
        parsed = parse_json_response(raw)
    except Exception as e:
        raise ValueError(f"Invalid JSON from pricing agent: {raw}") from e

    state["result"] = parsed
    return state


def build_pricing_graph():
    graph = StateGraph(PricingState)

    graph.add_node("fetch_prices", make_fetch_prices())
    graph.add_node("reason_price", pricing_reasoning)

    graph.set_entry_point("fetch_prices")
    graph.add_edge("fetch_prices", "reason_price")
    graph.add_edge("reason_price", END)

    return graph.compile()


pricing_graph = build_pricing_graph()


@app.post("/price", response_model=PriceResponse)
async def compute_price(req: PriceRequest):
    try:
        state = {
            "request": req.dict(),
            "price_map": {},
            "result": {}
        }
        out = await pricing_graph.ainvoke(state)
        return out["result"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/price/put")
async def put_price(item: PriceItem):
    """Admin endpoint: insert/update price"""
    await db.prices.update_one({"product_id": item.product_id}, {"$set": {"price": item.price}}, upsert=True)
    return {"ok": True, "product_id": item.product_id, "price": item.price}


@app.get("/price/{product_id}", response_model=PriceItem)
async def get_price(product_id: str):
    doc = await db.prices.find_one({"product_id": product_id})
    if not doc:
        raise HTTPException(status_code=404, detail="price not found")
    return PriceItem(product_id=doc["product_id"], price=doc["price"])
