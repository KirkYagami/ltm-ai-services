from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field


app = FastAPI()


patients_store = []

class PatientCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=60)
    age: int = Field(..., ge=0, le=130)
    phone: str = Field(..., min_length=10, max_length=15)
    email: str = Field(..., min_length=5, max_length=100)
    complaint: str = Field(..., min_length=3, max_length=200)


class PatientRead(BaseModel):
    id: str
    name: str
    age: int
    phone: str
    email: str
    complaint: str
    status: str


def find_patient(patient_id:str)->dict | None:

    for patient in patients_store:
        if patient["id"]==patient_id:
            return patient
    return None

def find_patient_by_email(email:str)->dict|None:
    for patient in patients_store:
        if patient["email"]==email:
            return patient
    return None


@app.post("/patients", response_model=PatientRead, status_code=status.HTTP_201_CREATED)
def register_patient(payload: PatientCreate, response: Response)->dict:

    existing_patient = find_patient_by_email(payload.email)
    if existing_patient:
        raise HTTPException(
            status_code =400,
            detail=f"Patient with email '{payload.email}' already registered",
            headers={"X-Error-Code": "DUPLICATE_EMAIL"},
        )


    # Generate sequential ID
    patient_id = f"P-{len(patients_store) + 1:03d}"

    patient = {
        "id": patient_id,
        "name": payload.name,
        "age": payload.age,
        "phone": payload.phone,
        "email": payload.email,
        "complaint": payload.complaint,
        "status": "registered",
    }

    patients_store.append(patient)

    # Custom success header
    response.headers["X-Patient-Id"] = patient_id

    return patient



@app.get("/patients", response_model=list[PatientRead])
def list_patient(status:str = Query(default=None))->list[dict]:

    if status is None:
        return patients_store
    
    return [
        patient for patient in patients_store if patient["status"]==status
    ]


@app.get("/patients/{patient_id}", response_model=PatientRead)
def get_patient(patient_id:str)->dict:
    patient = find_patient(patient_id=patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found."
        )

    return patient


@app.put("/patients/{patient_id}/discharge", response_model = PatientRead)
def discharge_patient(patient_id:str)->dict:
    patient = find_patient(patient_id=patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found."
        )
    if patient["status"]=="discharged":
        raise HTTPException(
            status_code=400,
            detail=f"Patient {patient_id} is already discharged.",
            headers={"X-Error-Code": "ALREADY_DISCHARGED"}
        )

    patient["status"] = "discharged"
    return patient

@app.delete("/patients/{patient_id}")
def delete_patient(patient_id:str)->dict:
    patient = find_patient(patient_id)

    if patient is None:
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found." 
        )
    patients_store.remove(patient)

    return {
        "deleted": patient_id,
        "status": "removed"
    }   

    
















