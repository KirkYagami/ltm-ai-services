from typing import Optional, List, Literal

from fastapi import FastAPI, Response, Path, Query
from pydantic import BaseModel, Field


app = FastAPI()


movies_store = []

movie_counter = 0

class MovieCreate(BaseModel):
    title: str = Field(
        min_length=1,
        max_length=100
    )

    genre: str = Field(
        min_length=3,
        max_length=30
    )

    runtime_min: int = Field(
        ge=1,
        le=600
    )

    watched: bool = False

class MovieRead(BaseModel):
    id: str
    title: str
    genre: str
    runtime_min: int
    watched: bool


# enpooint 1
@app.post("/movies", response_model=MovieRead, status_code=201)
def create_movie(movie:MovieCreate, response:Response)->dict:
    global movie_counter

    movie_counter +=1
    movie_id = f"M-{movie_counter:03d}"

    record = {
        "id": movie_id,
        "title": movie.title,
        "genre": movie.genre,
        "runtime_min": movie.runtime_min,
        "watched": movie.watched
    }

    movies_store.append(record)

    response.headers["X-Movie-Id"] = movie_id
    return record


# enpoint 2 -> list movies\
@app.get("/movies", response_model=list[MovieRead])
def list_movies(
    genre: Optional[str] = Query(default=None),
    watched_only: Optional[bool] = Query(default=None),
    search: Optional[str] = Query(default=None),
    sort: Optional[Literal["runtime_min", "title"]] = Query(default=None)
)-> list[dict]:

    result = movies_store.copy()

    if genre is not None:
        result = [movie
            for movie in result if movie["genre"]==genre
        ]

        # Filter by watched status
    if watched_only is not None:
        result = [
            movie
            for movie in result
            if movie["watched"] == watched_only
        ]

        # Search title, case-insensitive
    if search is not None:
        search_lower = search.lower()

        result = [
            movie
            for movie in result
            if search_lower in movie["title"].lower()
        ]

        # Sort
    if sort == "runtime_min":
        result.sort(key=lambda movie: movie["runtime_min"])

    elif sort == "title":
        result.sort(key=lambda movie: movie["title"])

    return result


# endpoinnt 3 -> get a specific move

@app.get("/movies/{movie_id}")
def get_movie(movie_id:str = Path(min_length=3))->dict:
    for movie in movies_store:
        if movie["id"] == movie_id:
            return movie

    return {
        "message": f"Movie '{movie_id}' not found"
    }

# endpint 4 -> update a movie

@app.put("/movies/{movie_id}")
def update_movie(movie:MovieCreate, movie_id:str = Path(min_length=3)):
    for existing_movie in movies_store:
        if existing_movie["id"] == movie_id:
            existing_movie["title"] = movie.title
            existing_movie["genre"] = movie.genre
            existing_movie["runtime_min"] = movie.runtime_min
            existing_movie["watched"] = movie.watched

            return existing_movie
    return {
        "message": f"Movie: {movie_id} not found."
    }


# enpoint 4 -> delete a movie
@app.delete("/movies/{movie_id}")
def delete_movie(movie_id:str = Path(min_length=3)):
    
    for index, movie in enumerate(movies_store):
        if movie["id"] == movie_id:
            movies_store.pop(index)
            return {
                "deleted": movie_id,
                "status": "removed"
            }
    return {
        "message": f"Movie: {movie_id} not found."
    } 







