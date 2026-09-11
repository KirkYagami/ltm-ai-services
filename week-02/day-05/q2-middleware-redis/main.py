from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware import LoggingMiddleware
from routes import router


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(LoggingMiddleware)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": "Weather Forecast API is running"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )