# Redis User Score Cache API — Instructions

## 1. Build Redis

```bash
cd ~/project/workspace/Project/redis-7.4.2
make
```

## 2. Start Redis

Keep this terminal open:

```bash
src/redis-server
```

Redis should run on:

```text
localhost:6379
```

## 3. Verify Redis

Open a **new terminal**:

```bash
cd ~/project/workspace
Project/redis-7.4.2/src/redis-cli ping
```

Expected output:

```text
PONG
```

## 4. Run Tests

From the workspace root:

```bash
cd ~/project/workspace
./pytest/run.sh
```

Or:

```bash
pytest -v
```

Expected result:

```text
6 passed
```

## 5. Run the FastAPI Application

Open another terminal:

```bash
cd ~/project/workspace/Project
python3 main.py
```

The API will be available at:

```text
http://localhost:8000
```

## 6. Test the API

### Root Endpoint

```bash
curl http://localhost:8000/
```

### Get User Score

```bash
curl http://localhost:8000/api/v1/users/10/score
```

First request:

```json
{
  "source": "computed",
  "data": {
    "user_id": 10,
    "score": 100,
    "status": "active"
  }
}
```

Second request should use Redis cache:

```json
{
  "source": "cache",
  "data": {
    "user_id": 10,
    "score": 100,
    "status": "active"
  }
}
```

### Clear User Cache

```bash
curl -X DELETE http://localhost:8000/api/v1/users/10/cache
```

Expected status:

```text
204 No Content
```

## 7. Check Redis Cache

Check the cached value:

```bash
Project/redis-7.4.2/src/redis-cli GET user:score:10
```

Check the TTL:

```bash
Project/redis-7.4.2/src/redis-cli TTL user:score:10
```

The initial TTL should be approximately:

```text
60
```

## 8. Important

Keep the Redis server running while running the tests or FastAPI application.

The workflow is:

Terminal 1:
    Redis Server
        ↓
    localhost:6379

Terminal 2:
    pytest / FastAPI
        ↓
    Redis Cache