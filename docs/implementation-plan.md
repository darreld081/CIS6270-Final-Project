# Base Unguided Masked Discrete Flow Matching — K562 DNA

Plan for Stage 1 of the CIS6270 final project: an unconditional masked-DFM
generator for 200 bp K562 regulatory DNA. This is the frozen denoiser that every
later sampler comparison (motif guidance, joint jumps, clamp, overwrite) reuses,
so the priority is correctness and reproducibility, not sample-quality tricks.

**Scope note.** This produces sequences whose *statistics* match held-out K562
DHS regions. It does not establish regulatory function; the project description
puts biological validation out of scope.

## Decisions already fixed

| Decision | Choice | Why |
|---|---|---|
| Compute | Colab / cloud GPU, with a CPU smoke path | No CUDA in the local WSL2 venv |
| Backbone | Residual dilated CNN (`models/base_models.py:CNNModel`) | Matches §4.3; DNABERT-2's BPE tokenizer cannot express per-base masking |
| Probability path | Mask path only (`x_0 = M^L`) | Required by the §4.2 motif-jump sampler |
| Configuration | **Constants at the top of each script.** No config framework. | A single dataset and one model size; a config tree costs more than it saves |

### What this changes in the existing code

- `models/dfm.py` — `DNAModel` (DNABERT-2 + DiT) is unused in Stage 1.
  `DFM.general_step` samples `x_0 = torch.randint_like(x_1, high=V)`, which is
  the *uniform* source; it must become the mask path.
- `models/base_models.py` — `CNNModel` needs an input embedding of size **5**
  (A/C/G/T/M) while output logits stay **4**. Its dilation schedule already lands
  exactly on L=200 (`200 + 512 - 64*8 = 200`), so no padding changes.
- `self.path` / `self.loss_fn` in `DFM` look like Meta's `flow_matching`
  library, which is not installed. The mask path and masked CE are ~30 lines; we
  write them directly rather than add the dependency.

## Layout

```
CIS6270-Final-Project/
├── data/
│   ├── raw/                  # downloaded, treated as untrusted
│   └── processed/            # {train,val,test}.npy + manifest.json
├── models/                   # original starter code; consumed by step 4
├── src/dfm/
│   ├── alphabet.py           # ids, encode/decode, revcomp, canonical   [done]
│   ├── data.py               # download, filter, split                  [step 2]
│   ├── path.py               # mask path + masked cross-entropy         [step 3]
│   ├── model.py              # CNNDenoiser, 5 in / 4 out                [step 4]
│   ├── module.py             # MaskedDFM LightningModule                [step 5]
│   ├── sampler.py            # unguided Euler sampler                   [step 6]
│   └── metrics.py            # GC, k-mer JSD, diversity                 [step 7]
├── scripts/
│   ├── prepare_data.py
│   ├── train.py              # hyperparameters as constants at the top
│   ├── sample.py
│   └── evaluate.py
├── tests/
└── runs/                     # checkpoints, sequences, metrics (gitignored)
```

Dependencies: `torch`, `pytorch-lightning`, `numpy`, `pandas`, `matplotlib`,
`tqdm`, plus `pytest`.

## Core components

### 1. Alphabet (`src/dfm/alphabet.py`) — done

```
A=0  C=1  G=2  T=3  M=4      VOCAB_SIZE=5   NUM_BASES=4   MASK_ID=4
```

The mask sits last so model output is always `logits[..., :4]` with no index
remapping. Provides `encode`/`decode`/`encode_many`/`decode_many`, `is_valid`,
`revcomp`, `revcomp_ids`, and `canonical` (the strand-agnostic grouping key).

### 2. Data (`src/dfm/data.py`, `scripts/prepare_data.py`)

Source: `data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt` from the
[DNA-Diffusion repo](https://github.com/pinellolab/DNA-Diffusion) (~12k sequences
per cell type, tab-separated). Inspect the real header before hardcoding column
names. If 12k K562 rows under-train the denoiser, that repo's
`data/master_dataset_and_filter.py` rebuilds the full ~3.59M-row DHS table.

Steps, in order:

1. Filter to K562 rows.
2. Uppercase; keep only length-200 strings over `{A,C,G,T}` (drop any `N`).
3. Group by `canonical(seq)` so exact duplicates and reverse-complement pairs
   stay together.
4. Assign each **group** to a split by chromosome: `chr1` → test, `chr2` → val,
   rest → train. A group spanning splits goes to the most-held-out split it
   touches (test > val > train), so no group crosses.
5. Write `{train,val,test}.npy` as `int8[N, 200]` plus `manifest.json` recording
   the source SHA-256, row counts after each filter, and per-split sizes — the
   §A.1 reproducibility record.

Downloaded files land in their own directory and are parsed with an explicit
column list; never `exec`/`pickle` on downloaded content.

12k × 200 int8 is 2.4 MB, so tensors live in memory: `TensorDataset` +
`DataLoader`. Batches yield `x_1: int64[B, 200]` only — no label.

### 3. Path and loss (`src/dfm/path.py`)

Eq. (2), each base revealed independently with probability `t`:

```python
def sample_x_t(x_1, t):                 # x_1: (B,L) int64, t: (B,) float
    keep = torch.rand(x_1.shape, device=x_1.device) < t[:, None]
    return torch.where(keep, x_1, MASK_ID)
```

`t ~ U(0, 1-1e-8)` per example, not per batch.

Eq. (7) — per-sample mean over masked positions, then batch mean:

```python
ce = F.cross_entropy(logits.transpose(1, 2), x_1, reduction="none")   # (B,L)
m  = (x_t == MASK_ID).float()
loss = ((ce * m).sum(1) / m.sum(1).clamp(min=1.0)).mean()
```

The `clamp(min=1.0)` is the `max(1, |M(x)|)` term: a fully-visible example must
contribute exactly zero, which is why it must not be `clamp(min=eps)`.

### 4. Denoiser (`src/dfm/model.py`)

`CNNModel` from `base_models.py` with `nn.Embedding(5, embed_dim)` (was 4) and a
final conv to **4** logits. Keep the residual dilated stack, GroupNorm, Swish,
and time conditioning via `GaussianFourierProjection` + per-block `Dense`
broadcast. Keep or drop the `out - out.mean(-1)` logit centering — it is
shift-invariant under CE, so it only has to be *consistent* between train and
sample.

Width 256 / 5 blocks is 4.01M params and should train in tens of minutes on a
T4; width 128 is 1.00M and is the CPU smoke size. The receptive field is 697
positions, so every output sees the whole 200 bp sequence.

### 5. Training (`src/dfm/module.py`, `scripts/train.py`)

Hyperparameters are constants at the top of `scripts/train.py` — no config
framework, just names you can read and edit in one place:

```python
EMBED_DIM = 256; HIDDEN_DIM = 256
BATCH_SIZE = 256; LR = 3e-4; WEIGHT_DECAY = 0.01
EPOCHS = 200; WARMUP_FRAC = 0.1; GRAD_CLIP = 1.0; SEED = 0
```

`MaskedDFM(pl.LightningModule)` keeps the scaffolding already in
`models/dfm.py:DFM` — AdamW, linear warmup → cosine decay to 0.1× via
`LambdaLR` on `estimated_stepping_batches`, `training_step` / `validation_step`
logging. Three changes from that starter code:

- `general_step` uses the mask path, not `torch.randint_like` (uniform source).
- the batch is `x_1` alone, not `x_1, label`.
- hyperparameters arrive as plain keyword arguments, not a `cfg` object, and
  `save_hyperparameters()` records them into the checkpoint.

`train.py` wires it up: `pl.seed_everything(SEED, workers=True)`, a
`ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)` for the §4.3
checkpoint-selection criterion, `gradient_clip_val=GRAD_CLIP`, and a
`CSVLogger` into `runs/<name>/`. Lightning's `accelerator="auto"` covers both the
CPU smoke run and the Colab GPU without a code change.

Validation uses a **fixed** masking pattern per batch index, reseeded the same
way every epoch. With fresh times and masks each epoch the validation loss
carries enough sampling noise that `save_top_k=1` is partly a lottery; pinning
the pattern compares successive epochs on the same problem.

A step-capped run (`--smoke`) stops mid-epoch, so `val_check_interval` must be
below `max_steps` or validation never fires and no checkpoint is selected.

Deferred to step 7, once the sampler and metrics exist: a callback that every N
epochs samples a small batch and logs GC content and 4-mer JSD against val,
which catches a denoiser that is fitting loss while producing degenerate
sequences.

Each run writes `runs/<name>/` with the checkpoint, the hyperparameters, the
seed, and package versions (§A.4).

### 6. Sampler (`src/dfm/sampler.py`)

§4.4 with ρ=γ=0 — the base update on every site:

```python
x = full((B, L), MASK_ID)
for k in range(K):
    t_k = k / K
    d_k = (1 / K) / (1 - t_k)                       # = 1 at k = K-1
    p   = softmax(model(x, full((B,), t_k)), -1)    # (B,L,4)
    y   = Categorical(probs=p).sample()
    reveal = (torch.rand(B, L) < d_k) & (x == MASK_ID)
    x = torch.where(reveal, y, x)
```

One denoiser call per step; all draws use predictions from `x_k` and are applied
together. `rho` / `gamma` are accepted but **raise** unless zero, so the Stage 2
controller has to implement them rather than silently no-op. Budgets
`K ∈ {64, 128, 256}` in the pilot; two get frozen for the later
budget-sensitivity experiment.

`return_trace=True` records per-step time, reveal probability and masked
fraction — the §A.3 diagnostic, and what the mask-survival test checks against
`1 - t_k`.

### 7. Evaluation (`src/dfm/metrics.py`, `scripts/evaluate.py`)

Against held-out K562 (chr1), on completed sequences:

- **Validity** — length 200 over `{A,C,G,T}` (assert, not metric).
- **GC content** — mean *and spread*. `gc_std_ratio` below 1 means the model is
  under-dispersed: GC centred correctly but over too narrow a range, which a
  comparison of means alone hides.
- **k-mer JSD** — Jensen–Shannon divergence in bits for `k ∈ {3,4,6}`; fix one
  `k` after the pilot. Also compute it for uniform-random 200-mers — that is the
  number that makes the JSD interpretable.
- **Diversity** — unique fraction; mean normalized pairwise Hamming distance over
  a random subsample of pairs (full pairwise is O(N²)).
- **Memorization** — exact matches to any training sequence or its revcomp.
- **Cost** — denoiser calls per sample, wall-clock ms/sequence.

Keep these pure and array-in: Stage 2 reuses them verbatim and only adds motif
success plus the outside-the-motif-span variant.

### 8. Tests (`tests/`)

The §A.2 implementation checks, written now so Stage 2 inherits them:

- `test_alphabet.py` — round-trips, revcomp involution, canonical key. **done**
- `test_data.py` — no canonical group in two splits; all rows length 200 and
  in-alphabet.
- `test_path.py` — `P(x_t[i] == MASK) ≈ 1 - t` over many draws; `t=0` → all
  mask; `t→1` → all clean. Masked CE is exactly 0 on a fully-visible example.
- `test_sampler.py` — with a frozen dummy denoiser, empirical mask survival
  matches `1 - t_k` at every step; the final step leaves zero masks; output is
  length-200 and nucleotide-only.

`test_path` and `test_sampler` are the two that catch real bugs — a subtly wrong
reveal schedule trains and samples without ever erroring.

## Execution order

| # | Step | Deliverable | Status |
|---|---|---|---|
| 1 | `alphabet.py` + scaffolding | tests green | **done** |
| 2 | `data.py` + `prepare_data.py` | 9,769 / 878 / 1,321 train/val/test + `manifest.json` | **done** |
| 3 | `path.py` (path + loss) | path/loss/schedule tests green | **done** |
| 4 | `model.py` (adapt `CNNModel` 5→4) | `(2,200)` → `(2,200,4)`; overfits 8 seqs | **done** |
| 5 | `module.py` + `train.py` | CPU smoke run: 50 steps, val_loss 1.340 | **done** |
| 6 | `sampler.py` + `sample.py` | 64 valid sequences from the smoke checkpoint | **done** |
| 7 | `metrics.py` + `evaluate.py` | `metrics.json` + figure, run on smoke samples | **done** |
| 8 | Colab pilot run | trained checkpoint + pilot metric table | next |
| 9 | Freeze config; record §A.1/§A.4 | seeds, versions, checkpoint | |

Steps 1–7 run on CPU at width 128; only step 8 needs the GPU. The smoke path
exists so the pipeline is proven end-to-end before paying for a runtime.

## Success criteria for Stage 1

On held-out chr1 K562: validity 100%; GC content within a couple of points;
4-mer JSD clearly below the uniform-random baseline; unique fraction ≈ 1.0 with
no exact training matches; checkpoint, hyperparameters and split manifest frozen
and recorded. Only then does the denoiser get frozen for the sampler comparisons.

## Open questions for the pilot

- 12k K562 sequences may under-train a 6M-param CNN — decide after the pilot
  whether to rebuild the full DHS table.
- Whether to keep logit centering in the CNN output.
- Final `k` for the k-mer JSD, and the two denoiser-call budgets.

Sources: [DNA-Diffusion](https://pinellolab.github.io/DNA-Diffusion) ·
[Zenodo v3 record](https://zenodo.org/records/17420419)
