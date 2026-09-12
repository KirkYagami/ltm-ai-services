#!/bin/bash

cat > database.py <<'PY'
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "sqlite:///./myapp.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()
PY

cat > models.py <<'PY'
from sqlalchemy import Column, Integer, String

from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )

    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash = Column(
        String(255),
        nullable=False
    )

    role = Column(
        String(50),
        nullable=False
    )
PY

cat > security.py <<'PY'
from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext


SECRET_KEY = "my-super-secret-key"
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:
    return pwd_context.verify(
        plain_password,
        hashed_password
    )


def create_token(email: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "sub": email,
        "role": role,
        "exp": expire
    }

    return jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )


def verify_token(token: str) -> dict:
    return jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM]
    )
PY

cat > main.py <<'PY'
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
PY

cat > requirements.txt <<'TXT'
sqlalchemy
fastapi
uvicorn
python-jose
passlib
bcrypt<4.0.0
pytest
httpx
TXT

cat > tests.py <<'PY'
import os

if os.path.exists("myapp.db"):
    os.remove("myapp.db")

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


# Test Case 1: Signup user with valid role
def test_signup_success():
    response = client.post(
        "/signup",
        params={
            "email": "user1@test.com",
            "password": "password123",
            "role": "customer"
        }
    )

    assert response.status_code == 200
    assert "User registered" in response.json()["message"]


# Test Case 2: Signup duplicate user should fail
def test_duplicate_signup():
    client.post(
        "/signup",
        params={
            "email": "user2@test.com",
            "password": "password123",
            "role": "customer"
        }
    )

    response = client.post(
        "/signup",
        params={
            "email": "user2@test.com",
            "password": "password123",
            "role": "customer"
        }
    )

    assert response.status_code == 400


# Test Case 3: Signup with invalid role should fail
def test_invalid_role_signup():
    response = client.post(
        "/signup",
        params={
            "email": "user3@test.com",
            "password": "password123",
            "role": "invalid"
        }
    )

    assert response.status_code == 400


# Test Case 4: Login with valid credentials
def test_login_success():
    client.post(
        "/signup",
        params={
            "email": "user4@test.com",
            "password": "password123",
            "role": "customer"
        }
    )

    response = client.post(
        "/login",
        params={
            "email": "user4@test.com",
            "password": "password123"
        }
    )

    assert response.status_code == 200
    assert "access_token" in response.json()


# Test Case 5: Login with wrong password should fail
def test_login_wrong_password():
    client.post(
        "/signup",
        params={
            "email": "user5@test.com",
            "password": "password123",
            "role": "customer"
        }
    )

    response = client.post(
        "/login",
        params={
            "email": "user5@test.com",
            "password": "wrongpassword"
        }
    )

    assert response.status_code == 401


# Test Case 6: Access protected tickets route with valid token
def test_access_tickets_with_token():
    client.post(
        "/signup",
        params={
            "email": "user6@test.com",
            "password": "password123",
            "role": "customer"
        }
    )

    login_resp = client.post(
        "/login",
        params={
            "email": "user6@test.com",
            "password": "password123"
        }
    )

    token = login_resp.json()["access_token"]

    response = client.get(
        "/tickets",
        params={"token": token}
    )

    assert response.status_code == 200
    assert "Tickets retrieved" in response.json()["message"]


# Test Case 7: Access support-only route with support token should succeed
def test_support_access_allowed():
    client.post(
        "/signup",
        params={
            "email": "support@test.com",
            "password": "password123",
            "role": "support"
        }
    )

    login_resp = client.post(
        "/login",
        params={
            "email": "support@test.com",
            "password": "password123"
        }
    )

    token = login_resp.json()["access_token"]

    response = client.get(
        "/support/all-tickets",
        params={"token": token}
    )

    assert response.status_code == 200
    assert "All tickets visible to support" in response.json()["message"]


# Test Case 8: Access support route with non-support token should fail
def test_support_access_denied():
    client.post(
        "/signup",
        params={
            "email": "user7@test.com",
            "password": "password123",
            "role": "customer"
        }
    )

    login_resp = client.post(
        "/login",
        params={
            "email": "user7@test.com",
            "password": "password123"
        }
    )

    token = login_resp.json()["access_token"]

    response = client.get(
        "/support/all-tickets",
        params={"token": token}
    )

    assert response.status_code == 403


# Test Case 9: Access admin route with admin token should succeed
def test_admin_access_allowed():
    client.post(
        "/signup",
        params={
            "email": "admin@test.com",
            "password": "password123",
            "role": "admin"
        }
    )

    login_resp = client.post(
        "/login",
        params={
            "email": "admin@test.com",
            "password": "password123"
        }
    )

    token = login_resp.json()["access_token"]

    response = client.get(
        "/admin/users",
        params={"token": token}
    )

    assert response.status_code == 200
    assert "Admin user management access" in response.json()["message"]


# Test Case 10: Access protected route with invalid token
def test_invalid_token_rejected():
    response = client.get(
        "/admin/users",
        params={"token": "fake.jwt.token"}
    )

    assert response.status_code == 401
PY

echo "Project files created successfully:"
echo "  database.py"
echo "  models.py"
echo "  security.py"
echo "  main.py"
echo "  requirements.txt"
echo "  tests.py"