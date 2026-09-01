from fastapi import FastAPI

app = FastAPI()


# @ is the path operation decorator
# '/' refers to the root path (last part of URL)
# Path is also called an endpoint or route
# get is an HTTP method (operation)
@app.get('/')
def index():
    return 'Hello world'

# Handling another path
@app.get('/property')
def property():
    return 'This is the property page'

# Returning JSON instead of a simple string
@app.get('/movies')
def movies():
    return {'movie list': ['movie 1', 'movie 2', 'movie 3']}


@app.get('/property/{id}')
def get_property(id:int):
    return f"This is a page for property {id}"



### Real-world Example: E-commerce Product Detail

@app.get('/products/{product_id}')
def get_product_details(product_id: int):
    # In a real application, you would fetch the product from a database
    product = {
        "id": product_id,
        "name": f"Product {product_id}",
        "price": 29.99,
        "in_stock": True
    }
    return product




@app.get('/products/')
def list_products(min_price: float, max_price: float = 1000, sort_by: str = "price"):
    return {
        "min_price": min_price,
        "max_price": max_price,
        "sort_by": sort_by,
        "products": f"List of products filtered by price between {min_price} and {max_price}, sorted by {sort_by}"
    }


### Example: Social Media API


from typing import Optional

@app.get('/profiles/{username}/posts/{post_id}/comments/')
def get_post_comments(
    username: str,
    post_id: int,
    sort_by: str = "newest",
    filter_by: Optional[str] = None,
    page: int = 1,
    limit: int = 20
):
    return {
        "username": username,
        "post_id": post_id,
        "comments": f"Comments for post {post_id} by {username}",
        "sort_by": sort_by,
        "filter_by": filter_by,
        "pagination": {"page": page, "limit": limit}
    }