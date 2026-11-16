from fastapi.testclient import TestClient
from app import app, chunks

client = TestClient(app)

def test_health():
    assert client.get("/health").json()["status"] == "ok"

def test_ingest_and_ask():
    chunks.clear()
    text = "The campus library is open from 8 AM to 8 PM on weekdays. Students need an ID card to borrow books."
    assert client.post("/ingest/text", json={"name":"library.txt","text":text}).status_code == 200
    result = client.post("/ask", json={"question":"When is the campus library open?"})
    assert result.status_code == 200
    assert result.json()["sources"]
