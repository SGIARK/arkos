"""
This is the official ARK2.0 backend!
(WARNING: currently in alpha!)

Dependencies:
* datetime
* fastapi
* pydantic
* typing
* uuid

Global variables:
* app
* event_db
* user_db

Pydantic models:
* EventBase
* EventCreate
* Event
* ChatMessage
* ChatResponse

API routes:
* GET /
* POST /api/events
* GET /api/events
* GET /api/events/{event-id}
* PUT /api/events/{event_id}
* DELETE /api/events/{event_id}
* POST /api/chat/message
* GET /api/chat/suggestions

Helper functions:
* get_current_user()
* parse_command(message: str) -> tuple[str, dict]
"""

from typing import List, Optional
from datetime import datetime, date
import uuid
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="ARK Calendar API")

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---


class EventBase(BaseModel):
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime


class EventCreate(EventBase):
    pass


class Event(EventBase):
    id: str
    created_at: datetime
    updated_at: datetime


class ChatMessage(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    action: Optional[str] = None
    data: Optional[dict] = None


# --- In-memory storage ---
events_db = {}
users_db = {"test_user": {"id": "test_user", "events": []}}

# --- Helper functions ---


def get_current_user():
    """
    Simplified auth - always returns test user
    """
    #TODO: implement real authentication
    return "test_user"


def parse_command(message: str) -> tuple[str, dict]:
    """Simple command parser for chat messages"""
    message = message.lower()

    if "add" in message or "schedule" in message:
        return "create_event", {"action": "create_event"}
    elif "show" in message or "list" in message:
        return "list_events", {"action": "list_events"}
    elif "delete" in message or "remove" in message:
        return "delete_event", {"action": "delete_event"}
    else:
        return "unknown", {}


# --- API Routes ---


@app.get("/")
async def read_root():
    """
    Implements GET /
    """
    return {"message": "ARK Calendar API"}


# Event endpoints
@app.post("/api/events/", response_model=Event)
async def create_event(event: EventCreate, user: str = Depends(get_current_user)):
    """
    Implements POST /api/events
    Create the specified event, returns it as confirmation
    """
    event_id = str(uuid.uuid4())
    current_time = datetime.now()

    event_db = Event(
        id=event_id,
        title=event.title,
        description=event.description,
        start_time=event.start_time,
        end_time=event.end_time,
        created_at=current_time,
        updated_at=current_time,
    )

    events_db[event_id] = event_db.dict()
    users_db[user]["events"].append(event_id)

    return event_db


@app.get("/api/events/", response_model=List[Event])
async def list_events(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    user: str = Depends(get_current_user),
):
    """
    Implements GET /api/events
    Returns a list of events
    """
    user_events = users_db[user]["events"]
    events = [events_db[event_id] for event_id in user_events]

    if start_date:
        events = [e for e in events if e["start_time"].date() >= start_date]
    if end_date:
        events = [e for e in events if e["end_time"].date() <= end_date]

    return events


@app.get("/api/events/{event_id}", response_model=Event)
async def get_event(event_id: str, user: str = Depends(get_current_user)):
    """
    Implements GET /api/events/{event-id}
    Returns details of the specific event
    """
    if event_id not in events_db:
        raise HTTPException(status_code=404, detail="Event not found")
    return events_db[event_id]


@app.put("/api/events/{event_id}", response_model=Event)
async def update_event(
    event_id: str, event_update: EventCreate, user: str = Depends(get_current_user)
):
    """
    Implements PUT /api/events/{event-id}
    Changes the specified event
    """
    if event_id not in events_db:
        raise HTTPException(status_code=404, detail="Event not found")

    event_db = events_db[event_id]
    update_data = event_update.dict(exclude_unset=True)

    for field, value in update_data.items():
        event_db[field] = value

    event_db["updated_at"] = datetime.now()
    events_db[event_id] = event_db

    return event_db


@app.delete("/api/events/{event_id}")
async def delete_event(event_id: str, user: str = Depends(get_current_user)):
    """
    Implements DELETE /api/events/{event_id}
    Deletes the specified event
    """
    if event_id not in events_db:
        raise HTTPException(status_code=404, detail="Event not found")

    del events_db[event_id]
    users_db[user]["events"].remove(event_id)

    return {"message": "Event deleted"}


# Chat endpoints
@app.post("/api/chat/message", response_model=ChatResponse)
async def process_chat_message(
    message: ChatMessage, user: str = Depends(get_current_user)
):
    """
    Implements POST /api/chat/message
    Send a message to the AI assistant, and returns a response
    """
    intent, data = parse_command(message.message)

    responses = {
        "create_event": "I'll help you create a new event. What's the title and when would you like to schedule it?",
        "list_events": "Here are your upcoming events...",
        "delete_event": "Which event would you like to delete?",
        "unknown": "I'm not sure what you'd like to do. You can ask me to create, show, or delete events.",
    }

    return ChatResponse(
        response=responses[intent], action=data.get("action"), data=data
    )


@app.get("/api/chat/suggestions")
async def get_chat_suggestions():
    """
    Implements GET /api/chat/suggestions
    Get a list of suggestions
    """
    #TODO: make this dynamic???
    return {
        "suggestions": [
            "Add a meeting tomorrow at 2pm",
            "Show my events for today",
            "Delete my 3pm meeting",
        ]
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
