# mock_airbnb_server.py (all in one cell for Jupyter)

import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
import uvicorn

# In-notebook FastAPI app
airbnb_app = FastAPI(title="Mock Airbnb API")

# In-memory mock data
MOCK_LISTINGS: Dict[str, Dict[str, Any]] = {
    "A101": {
        "id": "A101",
        "title": "Cozy apartment in Paris",
        "price_per_night": 120.0,
        "amenities": ["WiFi", "Kitchen", "Washer"],
        "description": "A bright, cozy apartment near the Eiffel Tower.",
        # Has a working HTML page route below
        "url": "http://127.0.0.1:8001/rooms/A101",
    },
    "A102": {
        "id": "A102",
        "title": "Modern loft in Berlin",
        "price_per_night": 90.0,
        "amenities": ["WiFi", "Elevator"],
        "description": "Stylish loft in the heart of Berlin.",
        # Has a working HTML page route below
        "url": "http://127.0.0.1:8001/rooms/A102",
    },
    "A103": {
        "id": "A103",
        "title": "Beach house in Goa",
        "price_per_night": 200.0,
        "amenities": ["Sea view", "Pool", "Breakfast included"],
        "description": "Wake up to the sound of waves.",
        # Intentionally BROKEN URL (no corresponding route)
        "url": "http://127.0.0.1:8001/rooms/THIS_DOES_NOT_EXIST",
    },
}

@airbnb_app.get("/current_time")
def current_time_endpoint():
    """Return current server time in UTC."""
    now = datetime.utcnow()
    return {"current_time": now}

@airbnb_app.get("/search")
def search_endpoint(
    location: str = Query(...),
    # checkin: str = Query(...),
    # nights: int = Query(1),
    min_price: float = Query(0),
    max_price: float = Query(10000),
):
    """Search for listings based on location, check-in date, nights, and price range."""
    results = []
    for listing in MOCK_LISTINGS.values():
        price = listing["price_per_night"]
        if min_price <= price <= max_price and location.lower() in listing["title"].lower():
            results.append(
                {
                    "id": listing["id"],
                    "title": listing["title"],
                    "price_per_night": price,
                    "url": listing["url"],
                }
            )
    return {"results": results}

@airbnb_app.get("/listing/{listing_id}")
def listing_details_endpoint(listing_id: str):
    listing = MOCK_LISTINGS.get(listing_id)
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")
    return listing

# mock listing pages, used as the URLs to verify
@airbnb_app.get("/rooms/{listing_id}", response_class=HTMLResponse)
def listing_page(listing_id: str):
    listing = MOCK_LISTINGS.get(listing_id)
    if not listing:
        raise HTTPException(status_code=404, detail="Listing page not found")
    html = f"""
    <html>
      <head><title>{listing['title']}</title></head>
      <body>
        <h1>{listing['title']}</h1>
        <p>Price per night: {listing['price_per_night']}</p>
        <p>{listing['description']}</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)

def run_airbnb_server():
    """Run the FastAPI app with uvicorn in a background thread."""
    config = uvicorn.Config(
        airbnb_app,
        host="127.0.0.1",
        port=8001,
        log_level="info",
    )
    server = uvicorn.Server(config)
    server.run()

# Start the server in the background
server_thread = threading.Thread(target=run_airbnb_server, daemon=True)
server_thread.start()

print("Mock Airbnb API server started on http://127.0.0.1:8001 🚀")
