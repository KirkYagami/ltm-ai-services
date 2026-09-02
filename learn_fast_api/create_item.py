import json

from pydantic import BaseModel, Field
from fastapi import Response, status, FastAPI, HTTPException


app = FastAPI()


class Item(BaseModel):
    name: str = Field(min_length=3, max_length=50)
    price: float = Field(gt=0)
    tags: list[str] = Field(default_factory=list)

def load_items():
    try:
        with open("items.json", "r") as f:
            data =  json.load(f)

        return {
            int(k): Item(**v)
            for k, v in data.items()
        }
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_items(items):
    with open("items.json", "w") as f:
        json.dump({str(k): v.model_dump() for k, v in items.items()}, f, indent=4)


items = load_items()

@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: Item):
    item_id = max(items.keys(), default=0) + 1
    items[item_id] = item
    save_items(items)
    return {"item_id": item_id, "item": item.model_dump()}

@app.get("/items/{item_id}")
def get_item(item_id: int):
    item = items.get(item_id)
    if not item:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    return item

@app.put("/items/{item_id}")
def update_item(item_id: int, updated_item: Item):
    if item_id not in items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    items[item_id] = updated_item
    save_items(items)
    return {"item_id": item_id, "item": updated_item.model_dump()}

@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    if item_id not in items:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Item not found")
    del items[item_id]
    save_items(items)
    return {"message": "Item deleted successfully"}

# def add(a, b):
#     return a + b

# def subtract(a, b):
#     return a - b    

# class Person(BaseModel):
#     name: str
#     age: int    

# class PersonWithOptionalAge():
#     name: str
#     age: int


# my_objects = {
#      1: Person, 2: add, 3: subtract 
#  }


# print(my_objects[1](name="John", age=30))
# print(my_objects[2](5, 3))