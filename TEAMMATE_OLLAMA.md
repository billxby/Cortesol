# Use the Cortesol model

It works from anywhere while the host Mac and tunnel are online.

1. Put `.env.teammate` in the Cortesol folder.
2. Run:

```bash
set -a; source .env.teammate; set +a
make run-ui
```

Open <http://127.0.0.1:8000>. Keep `.env.teammate` private.
