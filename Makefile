# Cheat sheet for the vLLM model server. Run `make` to list the steps.
# Pick a preset with PRESET=<name>; LLM_MODEL in .env must match it.

PRESETS := qwen gemma llama qwen-mid qwen-weak
PRESET ?= qwen
API := http://localhost:8000/v1
# Homebrew install first, otherwise the install.sh venv.
VLLM ?= $(shell command -v vllm 2>/dev/null || echo $(HOME)/.venv-vllm-metal/bin/vllm)

ifeq ($(filter $(PRESET),$(PRESETS)),)
$(error PRESET must be one of: $(PRESETS))
endif

.DEFAULT_GOAL := help
.PHONY: help mac-install mac-serve docker-up docker-status docker-logs docker-down \
	models tool-check eval-smoke test

help: ## Show this cheat sheet
	@echo "Usage: make <target> PRESET=<name>   presets: $(PRESETS)   (default: $(PRESET))"
	@awk 'BEGIN {FS = ":.*## "} \
		/^##@/ {printf "\n%s\n", substr($$0, 5)} \
		/^[a-z-]+:.*## / {printf "  make %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "Windows: Docker Desktop (WSL 2 backend). Run make inside WSL, or the same"
	@echo "         'docker compose --profile <preset> ...' commands in PowerShell."

##@ macOS (Apple Silicon, GPU via vllm-metal) — steps 1-2, then "Check"
mac-install: ## 1. Install vllm-metal (brew; falls back to install.sh without a current Xcode)
	brew tap vllm-project/vllm-metal https://github.com/vllm-project/vllm-metal
	brew install vllm-project/vllm-metal/vllm-metal || \
		curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash

mac-serve: ## 2. Start the server in the foreground (Ctrl-C stops it)
	GLOO_SOCKET_IFNAME=lo0 $(VLLM) serve --config src/tool_eval/server/presets/metal/$(PRESET).yaml

##@ Linux / Windows (NVIDIA GPU, Docker Compose, driver >= 580) — steps 1-2, then "Check"
docker-up: ## 1. Build the image and start the preset in the background
	@if [ "$$(uname)" = Darwin ]; then echo "Docker has no GPU on macOS; use: make mac-serve PRESET=$(PRESET)"; exit 1; fi
	docker compose --profile $(PRESET) up -d --build

docker-status: ## 2. Wait until STATUS shows "healthy" (first run downloads weights)
	docker compose --profile $(PRESET) ps

docker-logs: ## Follow server logs
	docker compose --profile $(PRESET) logs -f

docker-down: ## Stop the preset (required before starting another one: same port)
	docker compose --profile $(PRESET) down

##@ Check (any platform, server running)
models: ## 3. List served models (expect the preset name)
	@curl -s $(API)/models | python3 -m json.tool | grep '"id"'

tool-check: ## 4. Send one tool-calling request (expect tool_calls with get_weather)
	@curl -s $(API)/chat/completions -H 'Content-Type: application/json' -d '{ \
		"model": "$(PRESET)", \
		"messages": [{"role": "user", "content": "What is the weather in Paris?"}], \
		"tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather", \
			"parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}], \
		"chat_template_kwargs": {"enable_thinking": false}}' \
	| python3 -c "import json, sys; m = json.load(sys.stdin)['choices'][0]['message']; print('tool_calls:', m.get('tool_calls')); print('content:', m.get('content'))"

eval-smoke: ## 5. One evaluation run against the server
	LLM_BASE_URL=$(API) LLM_MODEL=$(PRESET) uv run python -m tool_eval.main run \
		--only rag-2024h1 --levels level1 --repeats 1

test: ## Unit tests, lint, types (no server needed)
	uv run pytest -q && uv run ruff check . && uv run mypy src tests
