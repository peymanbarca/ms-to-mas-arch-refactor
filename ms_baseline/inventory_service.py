from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from pymongo import MongoClient, ReturnDocument
from pymongo.errors import PyMongoError
import time
import threading

app = FastAPI()
db_client = MongoClient("mongodb://user:pass1@localhost:27017/")
inventory_col = db_client["ms_baseline"]["inventory"]
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
    inventory_col.delete_many({})
    items: List[CartItem] = request["items"]
    for item in items:
        inventory_col.insert_one({"sku": item['sku'], "stock": item['stock']})


def inject_failure(req: ReservationReq):
    INJECT_DELAY, INJECT_DROP_RATE = 0, 0
    if req.delay and req.delay > 0:
        INJECT_DELAY = req.delay
    if req.drop and req.drop > 0:
        INJECT_DROP_RATE = req.drop

    # inject delay
    if INJECT_DELAY > 0:
        time.sleep(INJECT_DELAY)

    if INJECT_DROP_RATE > 0:
        import random
        if random.randint(0, 99) < INJECT_DROP_RATE:
            # simulate dropped request
            raise HTTPException(status_code=503, detail="simulated service drop")


@app.post("/reserve")
def reserve_stock(req: ReservationReq):
    if not req.items:
        raise HTTPException(status_code=400, detail="empty_cart_items")

    # optional drop injection: simulate network failure by returning 500 occasionally
    inject_failure(req)

    # ============================
    # ATOMIC PATH (Transaction)
    # ============================
    if req.atomic_update:
        try:
            with lock:
                results = []

                # Step 1: validate all items
                for item in req.items:
                    doc = inventory_col.find_one(
                        {"sku": item.sku}
                    )
                    if not doc or doc["stock"] < item.qty:
                        raise ValueError(f"Out of stock: {item.sku}")

                # Step 2: decrement all
                for item in req.items:
                    res = inventory_col.find_one_and_update(
                        {"sku": item.sku},
                        {"$inc": {"stock": -item.qty}},
                        return_document=ReturnDocument.AFTER
                    )
                    results.append({
                        "sku": item.sku,
                        "remaining": res["stock"]
                    })

            return {
                "order_id": req.order_id,
                "status": "reserved",
                "items": results
            }

        except Exception as e:
            return {
                "order_id": req.order_id,
                "status": "out_of_stock",
                "reason": str(e)
            }

    # ============================
    # NON-ATOMIC PATH (Stepwise)
    # ============================
    else:
        updated = []

        for item in req.items:
            doc = inventory_col.find_one({"sku": item.sku})
            if not doc or doc["stock"] < item.qty:
                return {
                    "order_id": req.order_id,
                    "status": "out_of_stock",
                    "failed_sku": item.sku
                }

            new_stock = doc["stock"] - item.qty
            inventory_col.update_one(
                {"sku": item.sku},
                {"$set": {"stock": new_stock}}
            )

            updated.append({
                "sku": item.sku,
                "remaining": new_stock
            })

        return {
            "order_id": req.order_id,
            "status": "reserved",
            "items": updated
        }
