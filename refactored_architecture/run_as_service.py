import uvicorn
from fastapi import FastAPI

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run("procurement_agent:app", host="0.0.0.0", port=8009, reload=True)
