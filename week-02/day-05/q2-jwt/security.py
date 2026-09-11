from datetime import datetime, timedelta, timezone

from jose import jwt
from passlib.context import CryptContext


# Hard-coded secret key as required by the assignment
SECRET_KEY = "my-super-secret-key"

# JWT signing algorithm
ALGORITHM = "HS256"

# Password hashing configuration
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


def hash_password(password: str) -> str:
    """
    Hash a plaintext password using bcrypt.
    """
    return pwd_context.hash(password)


def verify_password(
    plain_password: str,
    hashed_password: str
) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.
    """
    return pwd_context.verify(
        plain_password,
        hashed_password
    )


def create_access_token(email: str, role: str) -> str:
    """
    Create a JWT containing:
    - sub: user's email
    - role: user's role
    - exp: expiration time
    """

    expire = datetime.now(timezone.utc) + timedelta(minutes=60)

    payload = {
        "sub": email,
        "role": role,
        "exp": expire
    }

    token = jwt.encode(
        payload,
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return token


def decode_access_token(token: str) -> dict:
    """
    Decode and validate a JWT.

    jose.jwt.decode automatically validates:
    - signature
    - token structure
    - expiration

    Raises JWTError when the token is invalid or expired.
    """

    payload = jwt.decode(
        token,
        SECRET_KEY,
        algorithms=[ALGORITHM]
    )

    return payload