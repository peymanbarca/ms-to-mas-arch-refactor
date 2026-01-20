import os
import logging
import time
import uuid
import datetime
import httpx

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from httpx import AsyncClient
import json
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
import asyncio

######################
# [load_memory] → fetch_candidates → fetch_prices → rank → [update_memory] → END
#######################


logger = logging.getLogger("product_search_agent")
logging.basicConfig(level=logging.INFO)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "ms_baseline")
PORT = int(os.getenv("PORT", 8008))
PRICING_SERVICE_URL = os.getenv("PRICING_SERVICE_URL", "http://localhost:8002")

llm = ChatOllama(model="qwen2", temperature=0.0, reasoning=False)

app = FastAPI(title="Product Search Agent")

# DB client will be set on startup
db_client: Optional[AsyncIOMotorClient] = None
db = None

class ProductOut(BaseModel):
    sku: str
    name: str
    description: str

class ProductSearchResultItem(ProductOut):
    price: float
    score: float

class ProductCreate(BaseModel):
    sku: str
    name: str
    description: str

class ProductSearchResponse(BaseModel):
    query: str
    results: List[ProductSearchResultItem]


class UserMemory(BaseModel):
    user_id: str
    summary: str
    updated_at: datetime


class ProductSearchAgentState(TypedDict):
    query: str
    user_id: str
    memory_summary: Optional[str]
    candidates: List[Dict[str, Any]]
    prices: Dict[str, float]
    results: List[Dict[str, Any]]


async def load_user_memory(user_id: str) -> Optional[str]:
    doc = await db.user_memory.find_one({"user_id": user_id})
    return doc["summary"] if doc else None


async def save_user_memory(user_id: str, summary: str):
    await db.user_memory.update_one(
        {"user_id": user_id},
        {"$set": {"summary": summary, "updated_at": datetime.utcnow()}},
        upsert=True
    )

async def load_memory_node(
    state: ProductSearchAgentState
) -> ProductSearchAgentState:

    user_id = state.get("user_id")
    if not user_id:
        state["memory_summary"] = None
        return state

    summary = await load_user_memory(user_id)
    state["memory_summary"] = summary
    return state


async def update_memory_node(
    state: ProductSearchAgentState) -> ProductSearchAgentState:

    user_id = state.get("user_id")
    if not user_id:
        return state

    interaction = f"""
        User searched for: {state['query']}
        Top results: {[r['name'] for r in state['results'][:3]]}
        """

    prompt = f"""
        You are a memory summarization agent.
        
        Existing summary:
        {state.get("memory_summary", "None")}
        
        New interaction:
        {interaction}
        
        Update the summary concisely.
        """

    response = await asyncio.to_thread(llm.invoke, prompt)
    new_summary = response.text().strip()

    await save_user_memory(user_id, new_summary)
    return state


async def fetch_candidates_tool(state: ProductSearchAgentState) -> ProductSearchAgentState:
    q = state["query"]

    # Prefer text search
    cursor = db.products.find(
        {"$text": {"$search": q}},
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(10)

    docs = await cursor.to_list(length=10)

    # Fallback to regex (still deterministic DB logic)
    if not docs:
        docs = await db.products.find(
            {"name": {"$regex": q, "$options": "i"}}
        ).limit(10).to_list(length=10)

    state["candidates"] = docs
    return state


async def fetch_prices_tool(state: ProductSearchAgentState) -> ProductSearchAgentState:
    product_ids = [d["sku"] for d in state["candidates"]]

    prices = {}
    if not product_ids:
        state["prices"] = prices
        return state

    payload = {
        "items": [{"product_id": pid, "qty": 1} for pid in product_ids],
        "promo_codes": []
    }

    async with httpx.AsyncClient() as client:
        r = await client.post(f"{PRICING_SERVICE_URL}/price", json=payload, timeout=10)
        r.raise_for_status()
        jr = r.json()

    for it in jr.get("items", []):
        prices[it["product_id"]] = it["unit_price"]

    state["prices"] = prices
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


async def ranking_node(state: ProductSearchAgentState) -> ProductSearchAgentState:
    products = [
        {
            "sku": d["sku"],
            "name": d["name"],
            "description": d.get("description", ""),
            "db_score": d.get("score", 1.0)
        }
        for d in state["candidates"]
    ]

    prices = state["prices"]

    memory_block = (
        f"User preference summary:\n{state['memory_summary']}\n\n"
        if state.get("memory_summary")
        else ""
    )

    prompt = f"""
    You are a product search ranking agent.
    
    Task:
    - Fill the final result by matching each product and price by sku from the Prices and Products input  
    - Respect user preferences if provided
    - Return final result ONLY valid JSON with below schema:
    
    {memory_block}
    
    Schema:
    {{
        "results": [
          {{
            "sku": string,
            "name": string,
            "description": string,
            "price": number,
            "score": number
          }}
        ]
    }}

    
    Prices:
    {prices}
    
    Products:
    {json.dumps(products, indent=2)}
    """

    logger.info(f'LLM Call Prompt: {prompt}')
    response = await asyncio.to_thread(llm.invoke, prompt)

    raw_response = response.text()
    input_tokens = response.usage_metadata.get("input_tokens")
    output_tokens = response.usage_metadata.get("output_tokens")
    total_tokens = response.usage_metadata.get("total_tokens")
    logger.info(f'LLM Raw response: {raw_response}')
    print(f'LLM Raw response: {raw_response}')

    logger.info(f'LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
                f' total_tokens: {total_tokens}')
    print(f'LLM Token Metrics: input_tokens: {input_tokens}, output_tokens: {output_tokens},'
                f' total_tokens: {total_tokens}')

    try:
        results = parse_json_response(raw_response)
        assert isinstance(results["results"], list)
    except Exception as e:
        raise ValueError(f"Invalid result output: {raw_response}") from e

    state["results"] = results["results"]
    return state


def build_product_search_agent():
    graph = StateGraph(ProductSearchAgentState)

    graph.add_node("load_memory", load_memory_node)
    graph.add_node("fetch_candidates", fetch_candidates_tool)
    graph.add_node("fetch_prices", fetch_prices_tool)
    graph.add_node("rank", ranking_node)
    graph.add_node("update_memory", update_memory_node)

    graph.set_entry_point("load_memory")
    graph.add_edge("load_memory", "fetch_candidates")
    graph.add_edge("fetch_candidates", "fetch_prices")
    graph.add_edge("fetch_prices", "rank")
    graph.add_edge("rank", "update_memory")
    graph.add_edge("update_memory", END)

    return graph.compile()


search_graph = build_product_search_agent()


@app.post("/products")
async def create_product(p: ProductCreate):
    await db.products.insert_one(p.dict())
    return {"status": "created", "sku": p.sku}

@app.get("/search", response_model=ProductSearchResponse)
async def search_products(q: str = Query(...), user_id: Optional[str] = Query(None), limit: int = 5):
    state = {
        "query": q,
        "user_id": user_id,
        "candidates": [],
        "prices": {},
        "results": []
    }

    try:
        out = await search_graph.ainvoke(state)
        print(f'------------\n {out}')
        return ProductSearchResponse(
            query=q,
            results=out["results"][:limit]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
