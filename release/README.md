# Releasing deVision to the Hugging Face Hub

The GitHub repository is the source of truth; the Hub repository (`lukbit/devision`, private) holds what a release
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

Versions are model versions (`v0.2` = round `v12-ext`), separate from training rounds. Hub tag `v0.2` ↔ GitHub
tag `model-v0.2` (the code the release was built and verified with); the card installs the latest code and shows
`revision="v0.2"` for pinning the weights (without it, `devision.load("lukbit/devision")` takes the Hub's latest).
The card says only the version; `provenance.json` records the training round and commit.

## Steps (manual)

```bash
# 1. build the release folder from the checkpoint named in release.yaml (also writes results / manifest / smoke)
uv run python scripts/release/hf.py build release/v0.2
# 2. check it: files, hashes, card metadata, and the same answers as the source checkpoint on CPU
uv run python scripts/release/hf.py verify release/v0.2
# 3. commit release/v0.2/ and tag the code the card installs
git add release/v0.2 && git commit -m "Release v0.2" && git tag model-v0.2 && git push origin master model-v0.2
# 4. dry run, then upload (hf auth login first, with write access to the repo)
uv run python scripts/release/hf.py publish release/v0.2
uv run python scripts/release/hf.py publish release/v0.2 --push
# 5. optional: verify what the Hub serves
hf download lukbit/devision --revision v0.2 --local-dir /tmp/devision-v0.2
uv run python scripts/release/hf.py verify release/v0.2 --folder /tmp/devision-v0.2
```

`publish` runs `verify` first, creates the repository **private** if it does not exist, stops if the repository
is public (making it public is the project owner's decision, done by hand on the Hub), refuses to overwrite an
existing version tag, uploads the folder in one commit whose message names the GitHub tag and commit, and tags it.

A new release: copy `release/v0.2/` to `release/v0.3/`, change `release.yaml` (version, git_tag, round) and
`notes.md`, and run the steps again.
