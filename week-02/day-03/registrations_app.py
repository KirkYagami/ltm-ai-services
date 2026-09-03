import json

from fastapi import FastAPI, Response, status, Header, HTTPException
from pydantic import BaseModel, Field


app = FastAPI()

FILE_PATH = "/home/coder/project/workspace/Project/data/workshops.json"

with open(FILE_PATH, "r") as f:
    workshops = json.load(f)


registrations_store = []



# model 
class Registration(BaseModel):
   attendee_name: str = Field(min_length=2, max_length=60)
   email: str = Field(min_length=5, max_length=100)
   workshop_id: str
   experience_level: str = Field(default="beginner", max_length=15)


def registrations_for(workshop_id: str) -> list[dict]:

    return [registration
            for registration in registrations_store
            if registration["workshop_id"]==workshop_id
            ]


@app.get("/workshops")
def list_workshops():
    return workshops


@app.post("/registrations", status_code=status.HTTP_201_CREATED)
def submit_registration(registration: Registration, response:Response) -> dict:

    # REG-001,. REG-002
    registration_id = f"REG-{len(registrations_store)+1:03d}"

    record = {
        "registration_id": registration_id,
        "attendee_name": registration.attendee_name,
        "email": registration.email,
        "workshop_id": registration.workshop_id,
        "experience_level": registration.experience_level,

    }

    registrations_store.append(record)
    response.headers["X-Registration-Id"] = registration_id

    return record

@app.get("/workshops/{workshop_id}/registrations")
def list_registrations_for_workshop(workshop_id: str, x_admin_view: str = Header(default="public"))-> dict:
    registrations = registrations_for(workshop_id=workshop_id)

    return {
        "admin_view": x_admin_view,
        "workshop_id": workshop_id,
        "registrations": registrations
    }


@app.get("/workshops/{workshop_id}/seats-remaining")
def seats_remaining(workshop_id: str) -> dict:
    workshop = None

    for item in workshops:
        if item["id"] == workshop_id:
            workshop=item
            break
    # no workshop found
    if workshop is None:
        return {"message": "Not found"}
        # raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail="Item not found")

    registrations = registrations_for(workshop_id=workshop_id)

    registered = len(registrations)
    remaining = max(workshop["capacity"] - registered, 0)

    return {
        "workshop_id": workshop_id,
        "capacity": workshop["capacity"],
        "registered": registered,
        "remaining": remaining
        
    }




