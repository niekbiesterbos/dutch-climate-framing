# Running LOME on an HPC Cluster

## Table of Contents

1. [What does the pipeline produce?](#what-does-the-pipeline-produce)
2. [Requirements](#requirements)
3. [Directory structure](#directory-structure)
4. [Step 0: Download the LOME container](#step-0-download-the-lome-container)
5. [Step 1: Prerequisites — the overlay fix](#step-1-prerequisites--the-overlay-fix)
6. [Step 2: Tokenization](#step-2-tokenization)
7. [Step 3: SpanFinder](#step-3-spanfinder)
8. [Step 4: Entity Typing](#step-4-entity-typing)
9. [Running the full pipeline at once](#running-the-full-pipeline-at-once)
10. [Scaling to multiple documents](#scaling-to-multiple-documents)
11. [Cluster-specific instructions](#cluster-specific-instructions)
12. [Known issues and fixes](#known-issues-and-fixes)

---

## What does the pipeline produce?

For each input document you get:

- **Frames + triggers**: which word triggered the frame
- **Semantic roles per frame**: e.g. for `Attempt_suasion` (triggered by `verzoekt`):
  - `Speaker`: De Kamer [ORG.Government]
  - `Addressee`: de regering [ORG.Government]
  - `Content`: om bij ouderenmishandeling een drie keer hogere straf in te voeren
- **Entity type labels**: PER, ORG.Government, ORG.PoliticalOrganization.Party, VAL.Number.Number, etc.

The pipeline consists of three steps:

```
Raw text (.txt) → [Tokenizer] → [SpanFinder] → [Entity Typer] → .comm + entities.tsv + output.txt
```

---

## Requirements

- Access to an HPC cluster with **Apptainer** (formerly Singularity)
- A GPU node — see [Cluster-specific instructions](#cluster-specific-instructions) for compatibility
- ~30 GB of available storage for the container, overlay, and model cache

---

## Directory structure

Two variables are used throughout this document. Set them once at the start of your session:

```bash
export BASE=/path/to/directory/containing/lome.sif
export WORKDIR=/path/to/your/working/directory
```

For example, on Habrok (RUG):

```bash
export BASE=/scratch/user_number
export WORKDIR=/scratch/user_number/lome
```

The expected layout:

```
$BASE/
├── lome.sif                  # Downloaded in Step 0
└── lome_overlay.img          # Created in Step 1

$WORKDIR/
├── read_output.py            # Output reader script (see below)
├── run_pipeline.sh           # Pipeline script (see below)
├── input/                    # Put your .txt files here
│   ├── doc1.txt
│   └── ...
├── tmp_workdir/              # Intermediate files (auto-populated)
├── output/                   # Final output (auto-populated)
├── nltk_data/                # NLTK data (auto-populated on first run)
└── cache/                    # Model cache (auto-populated on first run)
```

Create the subdirectories:

```bash
mkdir -p $WORKDIR/{input,tmp_workdir,output,nltk_data,cache}
```

Place all your `.txt` files in `$WORKDIR/input` before running the pipeline. Each document to be processed must be in a separate `.txt` file.

---

## Step 0: Download the LOME container

```bash
apptainer pull $BASE/lome.sif docker://hltcoe/lome
```

This converts the Docker image from [Docker Hub](https://hub.docker.com/r/hltcoe/lome) to an Apptainer `.sif` file. This may take 10–20 minutes depending on your connection and cluster speed.

---

## Step 1: Prerequisites — the overlay fix

> **Why this is needed:** The base container was built with PyTorch 1.7.1 (CUDA 10.2), which does not support modern GPUs (L40S, A100, A30). We fix this by creating a writable Apptainer overlay that upgrades PyTorch inside the container without modifying the container image itself.
>
> **Skip this step only if you are running on a V100**, which is supported by the original container out of the box.

### Create the overlay

```bash
apptainer overlay create --size 25600 $BASE/lome_overlay.img
```

This creates a 25 GB writable overlay. This size is required — two PyTorch installations (~1.8 GB each) plus overhead need the space.

### Upgrade PyTorch for SpanFinder (Python 3.9 env)

```bash
apptainer exec --nv --no-home \
    --overlay $BASE/lome_overlay.img \
    $BASE/lome.sif \
    /opt/anaconda3/envs/spanfinder/bin/pip install torch==1.13.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117 \
    --no-deps
```

### Upgrade PyTorch for Entity Typer (Python 3.6 env)

> `torch==1.10.2` is the last version with Python 3.6 support.

```bash
apptainer exec --nv --no-home \
    --overlay $BASE/lome_overlay.img \
    $BASE/lome.sif \
    /opt/anaconda3/envs/hiertype/bin/pip install torch==1.10.2+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --no-deps
```

### Verify

```bash
apptainer exec --nv --no-home \
    --overlay $BASE/lome_overlay.img \
    $BASE/lome.sif \
    /opt/anaconda3/envs/spanfinder/bin/python -c \
    "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected output:

```
1.13.1+cu117
True
NVIDIA <your GPU>
```

---

## Step 2: Tokenization

This step converts raw `.txt` files into `.comm` files (Concrete Communications format) and bundles them into a zip for SpanFinder.

```bash
for f in $WORKDIR/input/*.txt; do
    name=$(basename "$f" .txt)
    apptainer exec --no-home \
        --bind $WORKDIR/input:/input \
        --bind $WORKDIR/tmp_workdir:/tmp/workdir \
        $BASE/lome.sif \
        /opt/anaconda3/envs/spanfinder/bin/python /opt/lome/create-single-section-comm.py \
        /input/${name}.txt \
        /tmp/workdir/${name}.comm
done

zip -j $WORKDIR/tmp_workdir/tokenized_comms.zip \
    $WORKDIR/tmp_workdir/*.comm
```

---

## Step 3: SpanFinder

SpanFinder identifies frames, triggers, and semantic roles (Agent, Patient, Addressee, etc.). This is the core step for agency and framing analysis.

```bash
apptainer exec --nv --no-home \
    --overlay $BASE/lome_overlay.img \
    --env NLTK_DATA=/nltk_data \
    --env ALLENNLP_CACHE_ROOT=/cache \
    --env TRANSFORMERS_CACHE=/cache \
    --env PYTHONPATH=/opt/spanfinder \
    --bind $WORKDIR/tmp_workdir:/tmp/workdir \
    --bind $WORKDIR/nltk_data:/nltk_data \
    --bind $WORKDIR/cache:/cache \
    $BASE/lome.sif \
    /opt/anaconda3/envs/spanfinder/bin/python /opt/spanfinder/scripts/predict_concrete.py \
    --device 0 \
    /tmp/workdir/tokenized_comms.zip \
    /tmp/workdir/spanfinder_comms.zip \
    /opt/models/spanfinder/model.tar.gz
```

Expected runtime on a modern GPU: ~1–2 minutes per document including model loading.

SpanFinder writes a zip. The Entity Typer expects a directory, so unzip before continuing:

```bash
unzip -oj $WORKDIR/tmp_workdir/spanfinder_comms.zip \
    -d $WORKDIR/tmp_workdir/
```

---

## Step 4: Entity Typing

The Entity Typer labels each identified entity span with a hierarchical type: PER, ORG.Government, ORG.PoliticalOrganization.Party, VAL.Number.Number, etc.

> **Note on `--event_fall_through True`:** The container does not include an event typer model. This flag skips event typing cleanly. It has no impact on frame and role extraction, which is handled entirely by SpanFinder.

```bash
apptainer exec --nv --no-home \
    --overlay $BASE/lome_overlay.img \
    --env ALLENNLP_CACHE_ROOT=/cache \
    --env TRANSFORMERS_CACHE=/cache \
    --env PYTHONPATH=/opt/hiertype \
    --bind $WORKDIR/output:/output \
    --bind $WORKDIR/tmp_workdir:/tmp/workdir \
    --bind $WORKDIR/cache:/cache \
    $BASE/lome.sif \
    /opt/anaconda3/envs/hiertype/bin/python /opt/hiertype/aux/scripts/gaia/run_uncached.py \
    --in_path /tmp/workdir \
    --out_path /output \
    --contextualizer_name "/opt/models/xlm-roberta-base" \
    --entity_tool_prefix "Span Finder" \
    --entity_typer_model_path /opt/models/hiertype/aida-finetuned \
    --entity_ontology_file /opt/lome/aida-m36-ontology.txt \
    --entity_ontology_mapping_file /opt/lome/aida-m36-mapping.txt \
    --event_fall_through True \
    --device cuda
```

---

## Running the full pipeline at once

After a successful run, `output/` contains:
- `*.comm` — annotated communication files with all frames, roles, and entity mentions
- `entities.tsv` — entity type labels per document
- `output.txt` — human-readable overview of all frames and roles, joined with entity types

### `read_output.py`

Save as `$WORKDIR/read_output.py`:

```python
from concrete.util import read_communication_from_file
import glob, os, csv

def load_entities_tsv(tsv_path):
    entities = []
    if not os.path.exists(tsv_path):
        return entities
    with open(tsv_path) as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) < 5:
                continue
            doc = row[0].split('/')[-1]
            start, end, text, etype = int(row[1]), int(row[2]), row[3], row[4]
            entities.append((doc, start, end, text, etype))
    return entities

def get_entity_type(entities, doc, start, end):
    for e_doc, e_start, e_end, e_text, e_type in entities:
        if e_doc == doc and e_start == start and e_end == end:
            return e_type
    return None

entities = load_entities_tsv("/output/entities.tsv")

with open("/output/output.txt", "w") as out:
    for path in sorted(glob.glob("/data/*.comm")):
        comm = read_communication_from_file(path)
        doc = os.path.basename(path).replace(".comm", "")

        out.write(f"\n=== {doc} ===\n")
        out.write(f"Text: {comm.text[:200]}\n\n")

        mention_map = {}
        if comm.entityMentionSetList:
            for ems in comm.entityMentionSetList:
                for m in ems.mentionList:
                    try:
                        span_start = m.tokens.tokenization.tokenList.tokenList[m.tokens.tokenIndexList[0]].textSpan.start
                        span_end = m.tokens.tokenization.tokenList.tokenList[m.tokens.tokenIndexList[-1]].textSpan.ending
                        etype = get_entity_type(entities, doc, span_start, span_end)
                    except (TypeError, IndexError):
                        etype = None
                    mention_map[m.uuid.uuidString] = {
                        "text": m.text,
                        "type": etype
                    }

        if comm.situationMentionSetList:
            for sms in comm.situationMentionSetList:
                for sit in sms.mentionList:
                    out.write(f"Frame: {sit.situationKind} | trigger: '{sit.text}'\n")
                    if sit.argumentList:
                        for arg in sit.argumentList:
                            if arg.entityMentionId:
                                uid = arg.entityMentionId.uuidString
                                info = mention_map.get(uid, {})
                                out.write(f"  {arg.role}: '{info.get('text', '?')}' [{info.get('type', '?')}]\n")
                    out.write("\n")

print("Done. Output written to output/output.txt")
```

### `run_pipeline.sh`

Save as `$WORKDIR/run_pipeline.sh`. **Edit the two variables at the top before running.**

```bash
#!/bin/bash
set -e

# =============================================
BASE=/path/to/directory/containing/lome.sif
WORKDIR=/path/to/your/working/directory
# =============================================

echo "=== Step 1: Tokenization ==="
for f in $WORKDIR/input/*.txt; do
    name=$(basename "$f" .txt)
    apptainer exec --no-home \
        --bind $WORKDIR/input:/input \
        --bind $WORKDIR/tmp_workdir:/tmp/workdir \
        $BASE/lome.sif \
        /opt/anaconda3/envs/spanfinder/bin/python /opt/lome/create-single-section-comm.py \
        /input/${name}.txt \
        /tmp/workdir/${name}.comm
done

zip -j $WORKDIR/tmp_workdir/tokenized_comms.zip \
    $WORKDIR/tmp_workdir/*.comm

echo "=== Step 2: SpanFinder ==="
apptainer exec --nv --no-home \
    --overlay $BASE/lome_overlay.img \
    --env NLTK_DATA=/nltk_data \
    --env ALLENNLP_CACHE_ROOT=/cache \
    --env TRANSFORMERS_CACHE=/cache \
    --env PYTHONPATH=/opt/spanfinder \
    --bind $WORKDIR/tmp_workdir:/tmp/workdir \
    --bind $WORKDIR/nltk_data:/nltk_data \
    --bind $WORKDIR/cache:/cache \
    $BASE/lome.sif \
    /opt/anaconda3/envs/spanfinder/bin/python /opt/spanfinder/scripts/predict_concrete.py \
    --device 0 \
    /tmp/workdir/tokenized_comms.zip \
    /tmp/workdir/spanfinder_comms.zip \
    /opt/models/spanfinder/model.tar.gz

unzip -oj $WORKDIR/tmp_workdir/spanfinder_comms.zip \
    -d $WORKDIR/tmp_workdir/

echo "=== Step 3: Entity Typing ==="
apptainer exec --nv --no-home \
    --overlay $BASE/lome_overlay.img \
    --env ALLENNLP_CACHE_ROOT=/cache \
    --env TRANSFORMERS_CACHE=/cache \
    --env PYTHONPATH=/opt/hiertype \
    --bind $WORKDIR/output:/output \
    --bind $WORKDIR/tmp_workdir:/tmp/workdir \
    --bind $WORKDIR/cache:/cache \
    $BASE/lome.sif \
    /opt/anaconda3/envs/hiertype/bin/python /opt/hiertype/aux/scripts/gaia/run_uncached.py \
    --in_path /tmp/workdir \
    --out_path /output \
    --contextualizer_name "/opt/models/xlm-roberta-base" \
    --entity_tool_prefix "Span Finder" \
    --entity_typer_model_path /opt/models/hiertype/aida-finetuned \
    --entity_ontology_file /opt/lome/aida-m36-ontology.txt \
    --entity_ontology_mapping_file /opt/lome/aida-m36-mapping.txt \
    --event_fall_through True \
    --device cuda

echo "=== Copying .comm files to output ==="
cp $WORKDIR/tmp_workdir/*.comm $WORKDIR/output/

echo "=== Step 4: Reading output ==="
apptainer exec --no-home \
    --overlay $BASE/lome_overlay.img \
    --bind $WORKDIR/tmp_workdir:/data \
    --bind $WORKDIR/output:/output \
    $BASE/lome.sif \
    /opt/anaconda3/envs/spanfinder/bin/python $WORKDIR/read_output.py

echo "=== Done. Results in $WORKDIR/output/ ==="
```

Run it:

```bash
bash $WORKDIR/run_pipeline.sh
```

---

## Scaling to multiple documents

The pipeline scales automatically. Place all `.txt` files in `input/` and run the pipeline — no changes to any command are needed. The tokenization loop handles all files, and SpanFinder and the Entity Typer process the full batch in one pass.

For large batches, submit via your cluster's job scheduler rather than running interactively. A minimal SLURM example:

```bash
#!/bin/bash
#SBATCH --job-name=lome
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --partition=

bash $WORKDIR/run_pipeline.sh
```

Adjust `--partition`, `--time`, and `--mem` according to your cluster's documentation and the size of your batch.

---

## Cluster-specific instructions

### RUG — Habrok

**GPU compatibility:**

| GPU | Compatible? | Notes |
|-----|-------------|-------|
| L40S | ✅ with overlay fix | Interactive nodes: `gpu1.hb.hpc.rug.nl`, `gpu2.hb.hpc.rug.nl` |
| A100 | ✅ with overlay fix | Available via `gpu` partition |
| V100 | ✅ no fix needed | sm_70, supported by original container |

**Interactive GPU access (no queue wait):**

```bash
ssh gpu1.hb.hpc.rug.nl
# or
ssh gpu2.hb.hpc.rug.nl
```

Use interactive nodes for testing. Submit large batches via `sbatch` with `--partition=gpu`.

**Storage:** Use a directory on your high-capacity storage (e.g. `/scratch/$USER`) as `BASE`, and create your `WORKDIR` inside it.

### VU — ADA

**GPU compatibility:**

| GPU | Compatible? | Notes |
|-----|-------------|-------|
| A30 | ✅ with overlay fix | sm_80, requires CUDA 11.x |
| RTX 2070 | ✅ no fix needed | sm_75, supported by original container |

If you land on an RTX 2070 node, Step 1 can be skipped.

For large jobs, consider requesting access to **Snellius** (national supercomputer via SURF, available to all Dutch universities). Snellius has A100s and significantly more GPU capacity than ADA.

Documentation: [VU ADA HPC](https://vu.nl/en/research/more-about/computing) | [Snellius](https://www.surf.nl/snellius)

---

## Known issues and fixes

### `GLIBC_2.28 not found` on import

**Cause:** A locally installed PyTorch in your home directory conflicts with the container's version.

**Fix:** Always include `--no-home` in all `apptainer exec` calls. This prevents your home directory from being mounted inside the container.

### `Read-only file system` errors

**Cause:** The container filesystem is read-only by default.

**Fix:** Use `--overlay` for steps that need to write inside `/opt/...`. Use `--bind` to mount your own data directories.

### `CUDA kernel version mismatch` / GPU not recognized

**Cause:** PyTorch 1.7.1 (CUDA 10.2) does not support sm_80 (A100), sm_86 (RTX 30xx), or sm_89 (L40S).

**Fix:** Complete Step 1. This upgrades PyTorch to 1.13.1 (CUDA 11.7) for SpanFinder and 1.10.2 (CUDA 11.3) for the Entity Typer.

### `No space left on device` during overlay install

**Cause:** Overlay too small. Two PyTorch installs take ~3.6 GB total.

**Fix:** Always create the overlay at 25 GB minimum: `apptainer overlay create --size 25600 lome_overlay.img`

### `can't open overlay for writing, currently in use by another process`

**Cause:** Another Apptainer process still holds the overlay open (e.g. a crashed session).

**Fix:**

```bash
fuser $BASE/lome_overlay.img
kill <PID>
```

### `NotADirectoryError: spanfinder_comms.zip`

**Cause:** The Entity Typer expects a directory as input, not a zip file.

**Fix:** Always unzip the SpanFinder output before running the Entity Typer. This is included in `run_pipeline.sh`.

### `FileNotFoundError: None/args.json`

**Cause:** The event typer model is not present in the container.

**Fix:** Add `--event_fall_through True`. This skips event typing without affecting frame or role extraction.

### `list index out of range` in Entity Typer

**Cause:** `--entity_tool_prefix` does not match the tool name stored in the `.comm` file.

**Fix:** Use `--entity_tool_prefix "Span Finder"` — capital S, capital F, with a space. You can verify the stored name with:

```bash
apptainer exec --no-home \
    --bind $WORKDIR/tmp_workdir:/tmp/workdir \
    $BASE/lome.sif \
    /opt/anaconda3/envs/spanfinder/bin/python - << 'EOF'
from concrete.util import read_communication_from_file
import glob
path = glob.glob("/tmp/workdir/*.comm")[0]
comm = read_communication_from_file(path)
for ems in comm.entityMentionSetList:
    print(ems.metadata.tool)
EOF
```
