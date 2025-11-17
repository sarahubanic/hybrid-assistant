from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import requests
import json
from typing import Optional
from pathlib import Path

DB_PATH = "data/mistral_bot.db"
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral:latest"

app = FastAPI(title="Mistral Chatbot (local)")


class Item(BaseModel):
    id: Optional[int] = None
    name: str
    content: str


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            content TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


@app.post("/items", response_model=Item)
def create_item(item: Item):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("INSERT INTO items (name, content) VALUES (?, ?)", (item.name, item.content))
    conn.commit()
    item_id = cur.lastrowid
    conn.close()
    return Item(id=item_id, name=item.name, content=item.content)


@app.get("/items")
def list_items():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, content FROM items ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [Item(id=r[0], name=r[1], content=r[2]) for r in rows]


@app.get("/items/{item_id}")
def get_item(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name, content FROM items WHERE id = ?", (item_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Item not found")
    return Item(id=row[0], name=row[1], content=row[2])


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE items SET name = ?, content = ? WHERE id = ?", (item.name, item.content, item_id))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if changed == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return Item(id=item_id, name=item.name, content=item.content)


@app.delete("/items/{item_id}")
def delete_item(item_id: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM items WHERE id = ?", (item_id,))
    conn.commit()
    changed = cur.rowcount
    conn.close()
    if changed == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"deleted": item_id}


class ChatRequest(BaseModel):
    message: str
    item_id: Optional[int] = None


@app.post("/chat")
def chat(req: ChatRequest):
    # Build a short prompt; include item content if requested
    prompt = "You are a helpful assistant."
    if req.item_id:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT name, content FROM items WHERE id = ?", (req.item_id,))
        row = cur.fetchone()
        conn.close()
        if row:
            prompt += f" Use the following item as context (name: {row[0]}): {row[1]}"
        else:
            prompt += " The referenced item was not found."
    prompt += f"\nUser: {req.message}\nAssistant:" 

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.7
    }

    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error contacting Ollama API: {e}")

    # Try to extract text from common response shapes
    reply_text = None
    try:
        data = resp.json()
        # common fields: 'output', 'result' -> 'output', or 'text'
        if isinstance(data, dict):
            if "output" in data:
                reply_text = data["output"]
            elif "result" in data and isinstance(data["result"], dict) and "output" in data["result"]:
                reply_text = data["result"]["output"]
            elif "text" in data:
                reply_text = data["text"]
            else:
                # fallback to full json
                reply_text = json.dumps(data)
        else:
            reply_text = str(data)
    except Exception:
        reply_text = resp.text

    return {"reply": reply_text}


class DirectoryRequest(BaseModel):
    path: str


class FileCreateRequest(BaseModel):
    path: str
    content: str = ""


@app.post("/create-directory")
def create_directory(req: DirectoryRequest):
    try:
        path = Path(req.path).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "message": f"Directory created: {path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/create-file")
def create_file(req: FileCreateRequest):
    try:
        path = Path(req.path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(req.content, encoding="utf-8")
        return {"status": "success", "message": f"File created: {path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {"message": "Mistral FastAPI CRUD Chatbot running"}
