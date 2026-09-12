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
