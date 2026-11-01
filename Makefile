MODEL ?= Qwen/Qwen2.5-3B-Instruct
PY ?= python3

.PHONY: install test build-hf up-vllm up-hf down bench-vllm bench-hf sweep sweep-dry report agent dashboard

install:            ## client, sweep and report deps
	$(PY) -m pip install -r requirements-dev.txt -r agent/requirements.txt

test:
	$(PY) -m pytest -q

build-hf:           ## build the HF baseline image used by the sweep
	docker build -t llm-bench-hf:latest servers/hf

up-vllm:            ## vLLM + Prometheus + Grafana
	docker compose --profile vllm up -d

up-hf:              ## HF baseline + Prometheus + Grafana
	docker compose --profile hf up -d --build

down:
	docker compose --profile vllm --profile hf --profile agent down

bench-vllm:         ## quick benchmark of the running compose vLLM server
	$(PY) -m bench.loadgen --base-url http://localhost:8000 --model $(MODEL) --label vllm_quick \
	  --concurrency 1 8 32 --num-prompts 96 --out results/quick/vllm_quick.json

bench-hf:
	$(PY) -m bench.loadgen --base-url http://localhost:8001 --model $(MODEL) --label hf_quick \
	  --concurrency 1 8 32 --num-prompts 48 --out results/quick/hf_quick.json

sweep: build-hf     ## full grid (stop the compose servers first: it needs the GPU)
	$(PY) -m bench.sweep configs/sweep.yaml

sweep-dry:
	$(PY) -m bench.sweep configs/sweep.yaml --dry-run

report:
	$(PY) -m bench.report --results results --out results/report

agent:              ## interactive LangGraph agent on the compose vLLM server
	OPENAI_BASE_URL=http://localhost:8000/v1 MODEL=$(MODEL) $(PY) agent/agent.py

dashboard:          ## regenerate the Grafana dashboard JSON
	$(PY) deploy/grafana/build_dashboard.py
