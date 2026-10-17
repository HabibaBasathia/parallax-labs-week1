from fastapi.testclient import TestClient

import src.api as api

# No "with" block, so the slow startup warm-up does not run in tests
client = TestClient(api.app)

FAKE_ANSWER = {
    "answer": "A fake answer.",
    "retrieved_chunks": ["fake chunk"],
    "out_of_domain": False,
    "retrieval_time_s": 0.01,
    "generation_time_s": 0.02,
    "total_time_s": 0.03,
}


# ---------- /health ----------

def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------- /search ----------

def test_search_returns_results(monkeypatch):
    fake = [{"id": "doc_0_chunk_0", "text": "A dog story", "distance": 0.5,
             "topic": "bird_happy_day", "sentiment": "positive"}]
    monkeypatch.setattr(api, "run_search", lambda query, top_k, where: fake)

    response = client.post("/search", json={"query": "dog"})
    assert response.status_code == 200
    assert response.json()["count"] == 1
    assert response.json()["results"][0]["text"] == "A dog story"


def test_search_passes_filters_to_database(monkeypatch):
    captured = {}

    def fake_run_search(query, top_k, where):
        captured["where"] = where
        return []

    monkeypatch.setattr(api, "run_search", fake_run_search)
    response = client.post(
        "/search",
        json={"query": "dog", "sentiment": "negative", "topic": "bird_happy_day"},
    )
    assert response.status_code == 200
    assert captured["where"] == {
        "$and": [{"sentiment": "negative"}, {"topic": "bird_happy_day"}]
    }


def test_search_blank_query_returns_400():
    response = client.post("/search", json={"query": "   "})
    assert response.status_code == 400


def test_search_invalid_sentiment_returns_400():
    response = client.post("/search", json={"query": "dog", "sentiment": "happy"})
    assert response.status_code == 400


def test_search_wrong_type_returns_422():
    response = client.post("/search", json={"query": "dog", "top_k": "abc"})
    assert response.status_code == 422


def test_search_missing_query_returns_422():
    response = client.post("/search", json={})
    assert response.status_code == 422


def test_search_internal_error_returns_500(monkeypatch):
    def broken(query, top_k, where):
        raise RuntimeError("database is down")

    monkeypatch.setattr(api, "run_search", broken)
    response = client.post("/search", json={"query": "dog"})
    assert response.status_code == 500


# ---------- /ask ----------

def test_ask_returns_answer(monkeypatch):
    monkeypatch.setattr(api, "answer_question", lambda question: FAKE_ANSWER)

    response = client.post("/ask", json={"question": "What happened?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "A fake answer."


def test_ask_blank_question_returns_400():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_ask_empty_string_returns_422():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_llm_error_returns_500(monkeypatch):
    failed = dict(FAKE_ANSWER, answer="Error: LLM call failed after multiple retries.")
    monkeypatch.setattr(api, "answer_question", lambda question: failed)

    response = client.post("/ask", json={"question": "What happened?"})
    assert response.status_code == 500


# ---------- filter builder ----------

def test_build_where_no_filters_is_none():
    assert api.build_where(None, None) is None


def test_build_where_single_filter():
    assert api.build_where("positive", None) == {"sentiment": "positive"}


def test_build_where_both_filters():
    assert api.build_where("positive", "t1") == {
        "$and": [{"sentiment": "positive"}, {"topic": "t1"}]
    }