from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from jose import JWTError

from database import Base, engine, SessionLocal
from models import User
from security import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token
)


# Create FastAPI application
app = FastAPI()


# Create database tables when the application starts/imports
Base.metadata.create_all(bind=engine)


# ----------------------------------------
# Database dependency
# ----------------------------------------

def get_db():
    """
    Create one database session for each request
    and close it when the request finishes.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


# ----------------------------------------
# POST /register
# ----------------------------------------

@app.post("/register")
def register_user(
    email: str,
    password: str,
    db: Session = Depends(get_db)
):
    # Check whether the email already exists
    existing_user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="User already exists"
        )

    # Hash password before storing it
    password_hash = hash_password(password)

    # Create new user
    new_user = User(
        email=email,
        password_hash=password_hash,
        role="user"
    )

    db.add(new_user)
    db.commit()

    return {
        "message": "User registered successfully"
    }


# ----------------------------------------
# POST /login
# ----------------------------------------

@app.post("/login")
def login_user(
    email: str,
    password: str,
    db: Session = Depends(get_db)
):
    # Find user by email
    user = (
        db.query(User)
        .filter(User.email == email)
        .first()
    )

    # User doesn't exist
    if user is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    # Password doesn't match
    if not verify_password(
        password,
        user.password_hash
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    # Create JWT containing email and role
    access_token = create_access_token(
        email=user.email,
        role=user.role
    )

    return {
        "access_token": access_token
    }


# ----------------------------------------
# GET /protected
# ----------------------------------------

@app.get("/protected")
def protected_resource(token: str):
    try:
        # Decode and validate JWT
        payload = decode_access_token(token)

    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )

    # Extract information from JWT
    email = payload.get("sub")
    role = payload.get("role")

    return {
        "message": "Access granted",
        "user": email,
        "role": role
    }


# ----------------------------------------
# Run application directly
# ----------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8080,
        reload=True
    )