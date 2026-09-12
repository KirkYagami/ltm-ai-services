from fastapi import Depends, FastAPI, HTTPException
from jose import JWTError
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from models import User
from security import (
    create_token,
    hash_password,
    verify_password,
    verify_token,
)


app = FastAPI(title="Secure Support Ticket API")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@app.post("/signup")
def signup(
    email: str,
    password: str,
    role: str,
    db: Session = Depends(get_db)
):
    if role not in {"customer", "support", "admin"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid role"
        )

    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User exists"
        )

    user = User(
        email=email,
        password_hash=hash_password(password),
        role=role
    )

    db.add(user)
    db.commit()

    return {
        "message": "User registered"
    }


@app.post("/login")
def login(
    email: str,
    password: str,
    db: Session = Depends(get_db)
):
    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if not user or not verify_password(
        password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    access_token = create_token(
        user.email,
        user.role
    )

    return {
        "access_token": access_token
    }


@app.get("/tickets")
def get_tickets(token: str):
    try:
        payload = verify_token(token)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    return {
        "message": "Tickets retrieved",
        "user": payload["sub"],
        "role": payload["role"]
    }


@app.get("/support/all-tickets")
def support_all_tickets(token: str):
    try:
        payload = verify_token(token)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    if payload["role"] not in {"support", "admin"}:
        raise HTTPException(
            status_code=403,
            detail="Support access only"
        )

    return {
        "message": "All tickets visible to support"
    }


@app.get("/admin/users")
def admin_users(token: str):
    try:
        payload = verify_token(token)
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )

    if payload["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access only"
        )

    return {
        "message": "Admin user management access"
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080
    )
