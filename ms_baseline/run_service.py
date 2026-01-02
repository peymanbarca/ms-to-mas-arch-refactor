import uvicorn
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run("pricing_service:app", host="0.0.0.0", port=8002, reload=True)
