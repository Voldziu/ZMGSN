#!/usr/bin/env bash
# Evaluate several presets one after another: start the server, run the evaluation, stop it.
# Called by `make eval-presets`; every setting comes from the environment (see the Makefile).
#
#   PRESETS   presets to run, in order (space-separated)
#   REPEATS   repeats per scenario and level
#   ONLY      scenario ids (space-separated); empty runs all scenarios
#   LEVELS    tool levels (space-separated)
#   VLLM      vllm-metal binary (macOS only)
#
# Results go to results/presets-<preset>.jsonl and run workspaces to workspace/preset-runs/<preset>/
# (one directory per preset, so presets never overwrite each other's reviews). The PDF and arXiv
# caches stay shared in workspace/ and are filled once (`prewarm`) before the first preset. Each
# server's log goes to its own logs/server-<preset>-<time>.log. Every model gets one warm-up
# request before its first run. Re-running resumes: finished runs with the same configuration
# hash are skipped, so delete the file to start a preset from scratch.
set -uo pipefail

API=http://localhost:8000/v1
STARTUP_TIMEOUT=1800
ALL_PRESETS="qwen gemma llama qwen-mid qwen-weak"
cd "$(dirname "$0")/../../.." || exit 1
mkdir -p logs results

for p in $PRESETS; do
  case " $ALL_PRESETS " in *" $p "*) ;; *) echo "Unknown preset: $p (presets: $ALL_PRESETS)"; exit 2 ;; esac
done
if curl -sf -m 3 "$API/models" >/dev/null; then
  echo "A server is already answering on $API; stop it first (one preset at a time)."
  exit 2
fi

server_pid=""
current=""
server_log=""
PDF_CACHE_DIR="${PDF_CACHE_DIR:-workspace/pdf_cache}"
ARXIV_CACHE_DIR="${ARXIV_CACHE_DIR:-workspace/arxiv_cache}"
export PDF_CACHE_DIR ARXIV_CACHE_DIR
if [ "$(uname)" = Darwin ]; then PRESET_DIR=src/tool_eval/server/presets/metal
else PRESET_DIR=src/tool_eval/server/presets/cuda; fi

start_server() {
  if [ "$(uname)" = Darwin ]; then
    GLOO_SOCKET_IFNAME=lo0 "$VLLM" serve --config "$PRESET_DIR/$1.yaml" \
      > "$server_log" 2>&1 &
    server_pid=$!
  else
    docker compose --profile "$1" up -d --build > "$server_log" 2>&1 || return 1
  fi
}

server_alive() {
  if [ -n "$server_pid" ]; then
    kill -0 "$server_pid" 2>/dev/null
  else
    [ "$(docker compose --profile "$1" ps --format '{{.State}}' 2>/dev/null)" = running ]
  fi
}

stop_server() {
  [ -z "$current" ] && return
  if [ -n "$server_pid" ]; then
    # SIGTERM, not SIGINT: a background job of a script starts with SIGINT ignored, and vllm only
    # installs its own handler once the HTTP server is up, so SIGINT is lost during startup.
    kill -TERM "$server_pid" 2>/dev/null
    local i
    for i in $(seq 15); do kill -0 "$server_pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$server_pid" 2>/dev/null; then
      pkill -KILL -P "$server_pid" 2>/dev/null
      kill -KILL "$server_pid" 2>/dev/null
    fi
    wait "$server_pid" 2>/dev/null
    server_pid=""
  else
    docker compose --profile "$current" logs --no-color >> "$server_log" 2>&1
    docker compose --profile "$current" down > /dev/null 2>&1
  fi
  current=""
}
trap 'stop_server; exit 130' INT TERM

wait_ready() {
  local t0=$SECONDS
  while (( SECONDS - t0 < STARTUP_TIMEOUT )); do
    curl -sf -m 3 "$API/models" >/dev/null && return 0
    server_alive "$1" || return 1
    sleep 5 & wait $!  # `wait` returns at once on a signal, so the trap runs without delay
  done
  return 1
}

# One tool-calling request, so kernel compilation and caches are warm before the first run.
warm_up() {
  curl -sf -m 300 "$API/chat/completions" -H 'Content-Type: application/json' -d '{
    "model": "'"$1"'",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "chat_template_kwargs": {"enable_thinking": false}, "max_tokens": 64}' > /dev/null
}

# shellcheck disable=SC2086  # ONLY is intentionally word-split into arguments
uv run python -m tool_eval.main prewarm ${ONLY:+--only $ONLY} \
  || echo "prewarm failed; runs will fill the caches as they go"

failed=()
count=$(wc -w <<< "$PRESETS" | tr -d ' ')
index=0
for p in $PRESETS; do
  index=$((index + 1))
  echo "===== $p ($(date +%T))"
  current=$p
  server_log="logs/server-$p-$(date +%Y%m%d-%H%M%S).log"
  t0=$SECONDS
  if ! start_server "$p" || ! wait_ready "$p"; then
    echo "$p: server did not start; see $server_log"
    stop_server
    failed+=("$p")
    continue
  fi
  echo "$p: ready after $(( SECONDS - t0 ))s"
  warm_up "$p" && echo "$p: warmed up" || echo "$p: warm-up request failed"

  out="results/presets-$p.jsonl"
  # shellcheck disable=SC2086  # ONLY and LEVELS are intentionally word-split into arguments
  # TOOL_EVAL_PRESET_POSITION feeds the model progress bar (shown only on a terminal).
  # TOOL_EVAL_PRESET_FILE identifies the served model in the configuration hash.
  LLM_BASE_URL=$API LLM_MODEL=$p TOOL_EVAL_PRESET_POSITION="$index/$count" \
    TOOL_EVAL_PRESET_FILE="$PRESET_DIR/$p.yaml" \
    uv run python -m tool_eval.main run \
    ${ONLY:+--only $ONLY} --levels $LEVELS --repeats "$REPEATS" \
    --output "$out" --runs-dir "workspace/preset-runs/$p" \
    || failed+=("$p")
  stop_server

  # The quality table of the report: runs, successes with their interval, score, step limit.
  uv run python -m tool_eval.main report "$out" | awk '/^## /{show = ($0 == "## Quality")} show'
done

echo "===== done ($(date +%T)); results: results/presets-<preset>.jsonl"
echo "all models: uv run python -m tool_eval.main report results/presets-*.jsonl --csv results/summary.csv"
if (( ${#failed[@]} )); then
  echo "failed: ${failed[*]}"
  exit 1
fi
