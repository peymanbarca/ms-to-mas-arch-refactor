from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import requests
from pymongo import MongoClient
import uuid
import time
import httpx

app = FastAPI()
ORDER_COLL = MongoClient("mongodb://user:pass1@localhost:27017/")["ms_baseline"]["orders"]
INVENTORY_SERVICE_URL = "http://127.0.0.1:8001/reserve"
CART_SERVICE_URL = "http://127.0.0.1:8003/cart/"
PRICING_SERVICE_URL = "http://127.0.0.1:8002"

http_client = httpx.AsyncClient(timeout=10.0)


class CartItem(BaseModel):
    sku: str
    qty: int = Field(1, gt=0)


class Cart(BaseModel):
    cart_id: str
    items: List[CartItem] = []


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


class OrderCreate(BaseModel):
    items: List[CartItem] = []
    cart_id: str
    final_price: float
    atomic_update: bool = False
    delay: float = 0.0
    drop: int = 0


@app.post("/clear_orders")
def clear_orders():
    ORDER_COLL.delete_many({})


@app.post("/cart/{cart_id}/checkout")
async def checkout_cart(cart_id: str):
    # 1. retrieve cart from cart service
    # 2. retrieve final price from pricing service
    # orchestrate order placement -> inventory reservation -> payment processing -> book shipment -> notify user

    try:
        cart_resp = requests.get(CART_SERVICE_URL + f'{cart_id}', timeout=10)
        if cart_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Cart service error")
        cart: Optional[Cart] = cart_resp.json()
        if cart is None:
            raise HTTPException(status_code=404, detail="Cart not found")
        cart_items = cart['items']
        try:
            price_payload = {"items": [{"product_id": item['sku'], "qty": item['qty']} for item in cart_items],
                             "promo_codes": []}
            resp = await http_client.post(f"{PRICING_SERVICE_URL}/price", json=price_payload, timeout=10)
            resp.raise_for_status()
            j_resp: PriceResponse = resp.json()
            final_price = j_resp['total']

            return create_order(OrderCreate(items=cart_items, cart_id=cart_id, final_price=final_price,
                                            atomic_update=True, delay=0.0, drop=0))

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def create_order(order: OrderCreate):
    order_id = str(uuid.uuid4())
    start_time = time.time()
    # Save order INIT
    ORDER_COLL.insert_one({"_id": order_id, "items": [{'sku': item.sku, 'qty': item.qty} for item in order.items],
                           "cart_id": order.cart_id, "status": "INIT",
                          "final_price": order.final_price})

    # Call inventory service
    try:
        reserve_resp = requests.post(INVENTORY_SERVICE_URL, json={"order_id": order_id,
                                                                  "items": [{'sku': item.sku, 'qty': item.qty} for item in order.items],
                                                                  "atomic_update": order.atomic_update,
                                                                  "delay": order.delay,
                                                                  "drop": order.drop},
                                     timeout=10)
        if reserve_resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Inventory service error")
        reserve_result = reserve_resp.json()
        # Update order status
        ORDER_COLL.update_one({"_id": order_id}, {"$set": {"status": reserve_result["status"]}})
    except Exception as e:
        # ORDER_DB.update_one({"_id": order_id}, {"$set": {"status": "error"}})
        raise HTTPException(status_code=500, detail=str(e))

    end_time = time.time()
    latency = end_time - start_time
    return {"order_id": order_id, "status": reserve_result["status"], "latency": latency}
