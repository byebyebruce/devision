# v0.3 release validation — 2026-10-14

- `uv run --no-sync pytest -q`: 190 passed (20.46 s); one existing FastAPI/Starlette deprecation warning.
- `uv run --no-sync pyright`: 0 errors, 0 warnings.
- `scripts/release/hf.py verify release/v0.3`: passed all file hashes, metadata and CPU smoke comparisons against the source checkpoint.
- Current PyPI release source (`v0.2.1`) was loaded from a separate temporary checkout with the current dependency environment. It loaded this v0.3 release on CPU and matched all six source smoke answers within 1e-4. This checks backward compatibility of the inference code, not every historical dependency combination.
- Separate CPU timing: the same 200-question fixture used for v0.2, FP32, four threads, three warm-up requests excluded; P50 174.9 ms, P95 190.0 ms. The raw result is `cpu_latency.json`. This is local latency, not a hardware-independent guarantee.
- ModelScope staging was built from the verified HF release. Platform manifests contain the same 18 files and differ only in README content (plus the manifest file itself).
- `model.safetensors`: 2,073,754,784 bytes, SHA256 `110fea95889947c373258c88a396ad3eec574971815ee437b9c7cc3df8a5d2d5`.
- The original acceptance verdict remains `candidate=false`; `adoption.md` records the owner's explicit override. No training or temperature refit was performed for release.

Upload receipts and remote verification are recorded under the ignored `runs/release/` directory after publishing.
