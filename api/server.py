from fastapi import FastAPI

from api.routes import router

app = FastAPI(title="RAG + BitNet", version="0.1.0")
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    print("RAG BitNet server started. Models load lazily on first use.")
