---
title: Adapt Service
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Deterministic HTTP JSON Service

FastAPI service implementing `choose` and `repair` operations for ML intervention selection and repair validation.

## API Endpoint

POST `/adapt`

### Operations:
- `choose`: Select the best intervention candidate based on policy constraints
- `repair`: Validate ML repair operations including token labeling, PEFT configuration, and checkpoint integrity

## Local Testing

```bash
pip install -r requirements.txt
python -m pytest test_main.py -v
```

## Local Development

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
