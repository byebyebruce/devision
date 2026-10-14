# Releasing deVision to Hugging Face and ModelScope

The GitHub repository is the source of truth; the Hub repository ([`lukbit/devision`](https://huggingface.co/lukbit/devision), public) holds what a release
produces. Nothing is edited by hand on the Hub: the model card is generated, so a hand edit would be overwritten
by the next release and `verify` reports it as a changed file.

## Layout

| Path | In git | What |
|---|---|---|
| `release/card_template.md` | yes | the model card's prose; `{{...}}` placeholders are filled by the build |
| `release/<version>/release.yaml` | yes | which checkpoint (round + stage), Hub repo, version tag, GitHub tag, licence, base models, datasets, CPU latency file |
| `release/<version>/notes.md` | yes | the version's "training" and "limitations" sections |
| `release/<version>/results.json` | yes, written by build | every number in the card, read from `runs/<round>/eval/` |
| `release/<version>/MANIFEST.json` | yes, written by build | size and SHA256 of every released file |
| `release/<version>/smoke.json` | yes, written by build | fixed requests on `examples/` pictures and the source checkpoint's answers |
| `runs/release/<version>/` | no | the built Hub repository (weights, configs, card, `LICENSE`, `provenance.json`, `evaluation/`) |

Versions are model versions (`v0.3` = `r15-vision`; `v0.2` = `v12-ext`), separate from training rounds and Python package versions. Hub tag `v0.3` ↔ GitHub tag `model-v0.3`. Normal usage loads the latest model without a revision argument. Model tags do not trigger PyPI publishing.
The card says only the version; `provenance.json` records the training round and commit.

## Steps (manual)

```bash
# 1. build the release folder from the checkpoint named in release.yaml (also writes results / manifest / smoke)
uv run python scripts/release/hf.py build release/v0.3
# 2. check it: files, hashes, card metadata, and the same answers as the source checkpoint on CPU
uv run python scripts/release/hf.py verify release/v0.3
# 3. commit release/v0.3/ and tag the code the card installs
git add release/v0.3 && git commit -m "Release model v0.3" && git tag model-v0.3 && git push origin master model-v0.3
# 4. dry run, then upload (hf auth login first, with write access to the repo)
uv run python scripts/release/hf.py publish release/v0.3
uv run python scripts/release/hf.py publish release/v0.3 --push
# 5. optional: verify what the Hub serves
hf download lukbit/devision --revision v0.3 --local-dir /tmp/devision-v0.3
uv run python scripts/release/hf.py verify release/v0.3 --folder /tmp/devision-v0.3
```

`publish` runs `verify` first, creates the repository **private** if it does not exist (making it public is the project
owner's decision, done by hand on the Hub), refuses to overwrite an
existing version tag, uploads the folder in one commit whose message names the GitHub tag and commit, and tags it.

Changing only the card of a released version (template or notes): `build`, `verify`, commit, then
`publish release/v0.3 --docs --push`. It uploads only `README.md`, `MANIFEST.json` and `evaluation/` to the main
branch; the weights and the version tag stay as they were, so `revision="v0.2"` keeps the card it was released with.

For a new version, create a new release directory with `release.yaml` and `notes.md`, then build and verify fresh artifacts. Do not reuse another checkpoint's CPU latency. Merge the reviewed release from `develop` into `master` before pushing the model tag; preserve unrelated working changes.


## ModelScope mirror

The existing repository is [`baibaibai000/devision`](https://modelscope.cn/models/baibaibai000/devision). Its id is recorded as `modelscope_repo` in the release config. The mirror uses the same weights, configuration, evaluation evidence and logo; only README download instructions differ. Its staging directory is always `runs/release/modelscope`, and `MANIFEST.modelscope.json` records its hashes. Authentication uses the existing ModelScope credential store; repository visibility is preserved.

```bash
uv run --with modelscope-hub python scripts/release/ms.py build release/v0.3
uv run --with modelscope-hub python scripts/release/ms.py publish release/v0.3
uv run --with modelscope-hub python scripts/release/ms.py publish release/v0.3 --push
```

The build verifies the HF release first. Publishing verifies the mirror against the source checkpoint on CPU, refuses an existing version tag, uploads it, checks remote sizes and SHA256s, creates the version tag and verifies the tagged files. The private upload receipt is in `runs/release/modelscope-upload.json`; never commit credentials. Both platform version tags preserve earlier releases.

For v0.3, `evaluation/adoption.md` records the owner's trade-off decision, while `evaluation/accept_vs_v0.2.json` preserves the failed prewritten acceptance verdict. Publication does not rewrite experiment evidence or refit temperatures.
