# ERS Pole Lab — web UI

React + Vite front end for the 2026 qualifying model, styled as a 70s printed race programme (light and dark).

- **Season** (`#/`): every 2026 round with its qualifying energy cap, the real pole, and the model's latest
  qualifying lap against it. Rounds without a racing line for their current layout can't run yet.
- **Lap** (`#/round/13`): the model's lap on a map coloured by what the MGU-K does, with speed (against the
  real pole lap), MGU-K power and state of charge. **Settings** explains the event's energy rules and sets the run.

Runs go through the API in `backend/server.py`, which starts `main.py` in the background and streams its log.
Jobs live in the server's memory, so restarting the server stops a run in progress.

```bash
./start.sh                  # API on :8000, UI on :5173, from the repository root
npm run dev                 # UI only; VITE_API_URL points it at another API (default http://localhost:8000)
npm run build && npm run lint
```
