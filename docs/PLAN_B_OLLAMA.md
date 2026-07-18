# Plan B: local Ollama proposer

This deployment replaces the Freesolo checkpoint with a schema-constrained
`qwen3.5:9b` proposal policy. It does **not** replace the Cortesol ledger. The
model still only proposes operations; screening, validation, belief arithmetic,
propagation, and audit remain deterministic.

## One-time setup

```bash
make plan-b-setup
```

This pulls the 6.6 GB base model and creates the local alias
`cortesol-proposer:plan-b` from `deploy/ollama/Modelfile`.

Create a local secret (the file is git-ignored):

```bash
python -c 'import secrets; print("CORTESOL_API_KEY=" + secrets.token_urlsafe(32))' > .env.plan-b
cat >> .env.plan-b <<'EOF'
CORTESOL_RUN_ID=cortesol-plan-b
CORTESOL_OLLAMA_MODEL=cortesol-proposer:plan-b
EOF
```

## Serve and expose

Terminal 1:

```bash
set -a; source .env.plan-b; set +a
make plan-b-serve
```

Terminal 2 (Cloudflare quick tunnel):

```bash
make plan-b-tunnel
```

The tunnel prints a public HTTPS URL. Keep both terminals running during the
demo. Ollama remains on `127.0.0.1`; only the authenticated, schema-locked
gateway is public.

Verify it:

```bash
curl https://YOUR-TUNNEL.trycloudflare.com/health
```

## Drop-in Freesolo client configuration

The gateway implements the response shape already consumed by
`FreesoloExtractor`. On the machine running the pipeline:

```bash
export FREESOLO_RUN_ID=cortesol-plan-b
export FLASH_API_URL=https://YOUR-TUNNEL.trycloudflare.com
export FREESOLO_API_KEY='the CORTESOL_API_KEY value from the serving Mac'
```

No pipeline code change is required for this remote mode. For a pipeline running
on the same Mac, bypass the public tunnel:

```bash
export CORTESOL_EXTRACTOR=ollama
export CORTESOL_OLLAMA_MODEL=cortesol-proposer:plan-b
make run-ui
```

## Security notes

- Never bind Ollama itself to `0.0.0.0` or tunnel port 11434. Its native local
  API has no authentication and exposes model-management endpoints.
- The gateway ignores caller-supplied system and assistant messages, requires a
  bearer token, limits input size/concurrency, forces the Cortesol JSON schema,
  validates provenance, and fails closed to `REJECT` on malformed generations.
- A quick-tunnel hostname changes when restarted. For a stable demo hostname,
  create a named Cloudflare Tunnel and point it at `http://127.0.0.1:8787`.
