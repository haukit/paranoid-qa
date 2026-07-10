.PHONY: install index graph-index demo-index demo-artifact serve health ready ask docker-build docker-run

install:
	uv sync --extra serve --extra dev

index:
	uv run python -c "from paranoid_qa.index import build_index; build_index()"

graph-index:
	uv run python -c "from paranoid_qa.aggregate import build_lightrag; build_lightrag()"

demo-index:
	rm -rf .storage .lightrag
	$(MAKE) index
	$(MAKE) graph-index

demo-artifact:
	mkdir -p demo_artifacts
	tar -czf demo_artifacts/ntsb-demo-index.tar.gz .storage .lightrag

serve:
	uv run uvicorn paranoid_qa.server:app --reload

health:
	curl -f http://localhost:8000/healthz

ready:
	curl -f http://localhost:8000/readyz

ask:
	curl -N -X POST http://localhost:8000/ask \
		-H "Content-Type: application/json" \
		-H "X-Demo-Session: $${PARANOID_QA_DEMO_SESSION:-}" \
		-d '{"question":"What is this corpus about?"}'

docker-build:
	docker build -t paranoid-qa:local .

docker-run:
	docker run --rm --env-file .env -p 8000:8000 paranoid-qa:local
