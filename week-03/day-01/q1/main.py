from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from jose import JWTError

from database import Base, SessionLocal, engine
from models import InventoryService
from security import (
    create_access_token,
    decode_token,
    hash_secret,
    verify_secret,
)


app = FastAPI(title="Secure Inventory Sync API")


# Create database tables when the module is imported.
Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.post("/register-service")
def register_service(
    service_id: str,
    service_secret: str,
    service_name: str,
    db: Session = Depends(get_db)
):
    existing_service = (
        db.query(InventoryService)
        .filter(InventoryService.service_id == service_id)
        .first()
    )

    if existing_service:
        raise HTTPException(
            status_code=400,
            detail="Inventory service already exists"
        )

    service = InventoryService(
        service_id=service_id,
        service_secret_hash=hash_secret(service_secret),
        service_name=service_name
    )

    db.add(service)
    db.commit()

    return {
        "message": "Inventory service registered"
    }


@app.post("/token")
def issue_token(
    service_id: str,
    service_secret: str,
    db: Session = Depends(get_db)
):
    service = (
        db.query(InventoryService)
        .filter(InventoryService.service_id == service_id)
        .first()
    )

    if not service or not verify_secret(
        service_secret,
        service.service_secret_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid service credentials"
        )

    access_token = create_access_token(service_id)

    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.get("/inventory/items")
def list_inventory_items(token: str):
    try:
        decode_token(token)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    return {
        "items": [
            {
                "sku": "SKU-1001",
                "name": "Wireless Mouse",
                "quantity": 120
            },
            {
                "sku": "SKU-1002",
                "name": "Mechanical Keyboard",
                "quantity": 45
            },
            {
                "sku": "SKU-1003",
                "name": "USB-C Hub",
                "quantity": 78
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )
