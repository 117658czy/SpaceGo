<div align="center">

# SpaceGo 维空弈

**A from-scratch AlphaZero-style Go engine · HMGS-DP search × CA-MOE mixture-of-experts**

C++17 · CUDA · LibTorch · GTP · ~50k lines of hand-written code

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](CMakeLists.txt)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](CMakeLists.txt)
[![LibTorch](https://img.shields.io/badge/LibTorch-2.6.0%2Bcu124-EE4C2C.svg)](CMakeLists.txt)
[![GTP](https://img.shields.io/badge/Protocol-GTP-4B8BBE.svg)](src/network/protocol/gtp_protocol.cpp)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[Quick start](#quick-start) · [Algorithms](#core-algorithms-and-innovations) · [Inference path](#inference-path-and-model-files) · [Training](#training-system) · [Benchmarks](#measured-results-and-current-status) · [Roadmap](#roadmap-and-contribution-areas)

[中文文档](README.md)

</div>

---

> **Status: research prototype.**
>
> The architecture, the training loop and the engineering chain all work end to end. Playing strength
> is still far below mainstream engines, and the search plus the feature encoding have known defects —
> all of them are documented below with their impact, instead of being hidden behind marketing copy.
>
> Measured: **−349.5 points** against KataGo at 10 visits, 35–55 s per move at 1600 simulations on a
> GTX 1660 Ti. If you want a strong opponent, use KataGo. If you want a complete engine stack you can
> read, build and audit, keep reading.

---

## Why it is worth a look

- **A complete engine stack, not a tuning script.** Bitboards, search, neural inference, a training
  loop, the GTP protocol, a Win32 GUI, unit tests and the build system are all written from scratch —
  roughly 50k lines of C++/CUDA across 139 files.
- **Two original algorithm skeletons.** HMGS-DP on the search side, CA-MOE on the inference side.
  They are decoupled through evaluator injection, so each can be replaced or benchmarked independently.
- **A reproducible training loop.** Self-play → experience replay → gated accept/rollback → KataGo
  score-gap evaluation → teacher distillation. Logs and checkpoint formats are preserved, so every
  number in this README can be traced back to a training run.
- **Engineering over paper-writing.** Trimmed CMake builds (CUDA / LibTorch / training / GUI / tests
  are independent switches), graceful degradation on failure, cross-platform abstraction, Chinese
  comments and 17 algorithm documents.
- **Honest benchmarks.** The repository records both what works and what does not, including the
  −349.5 point gap. No "industrial strength" claims.

> If this project is useful to you, a **Star** is what keeps the training logs, model releases and
> benchmark data public.

---

## Table of contents

- [Engine overview](#engine-overview)
- [Inference path and model files](#inference-path-and-model-files)
- [Core algorithms and innovations](#core-algorithms-and-innovations)
- [Training system](#training-system)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Measured results and current status](#measured-results-and-current-status)
- [Known limitations](#known-limitations)
- [Roadmap and contribution areas](#roadmap-and-contribution-areas)
- [Docs and license](#docs-and-license)

---

## Engine overview

| Dimension | Implementation |
|-----------|----------------|
| Language / standard | C++17 (engine, training, GUI) + CUDA C++ |
| Board representation | 19×19 bitboards (`6 × uint64_t` per color) + Zobrist hashing |
| Search | HMGS-DP: UCT + transposition-table DAG merging + batched inference; virtual loss and work-stealing queues are scaffolded, but the GTP main loop is currently a single-threaded batched implementation |
| Neural network | CA-MOE: shared trunk (10 residual blocks / 256 channels) + 8 policy heads / 8 value heads + gating |
| Model size | ~**62 million** parameters, ~**237 MB** per checkpoint |
| Input / output | `17 × 19 × 19` feature tensor → 362-dim policy (361 points + pass) + 1-dim value |
| Inference backend | LibTorch (CPU / CUDA), batched forward passes, mixed-precision training |
| Protocol | GTP (`name` / `version` / `boardsize` / `clear_board` / `komi` / `play` / `genmove` / `undo` / `showboard` / `final_score` / `time_settings`) |
| Training modes | Supervised pre-training / unsupervised self-reinforcement / diploid competition / ELO-gated online training |
| Engineering | Trimmable CMake build, GDI+ GUI, GoogleTest unit tests, 17 algorithm documents |
| License | AGPL-3.0 |

---

## Inference path and model files

This is the code path that actually drives GTP play, listed in call order:

```text
spacego.exe --gtp --model=<checkpoint>.pt --simulations=1600
        │
        ▼
src/main.cpp                     parses --gtp / --model= / --simulations=
        │                        RunGTP()
        ▼
src/decision/engine/go_engine.cpp
        ├─ GoEngine::initialize()            checks whether the model suffix is .pt
        ├─ GoEngine::init_neural_engine()    constructs CA_MOEExpertSystem(8, 19, 0.1, 256)
        │      ├─ CA_MOEExpertSystem::load() reads a LibTorch OutputArchive checkpoint
        │      ├─ to(CUDA) + set_eval_mode()
        │      ├─ policy_cb()                policy prior (362 dims)
        │      ├─ value_cb()                 value (from the side-to-move perspective)
        │      └─ batch_cb()                 batched policy + value
        ├─ HMGSDP_Optimized::set_neural_evaluator()
        ├─ HMGSDP_Optimized::set_batch_evaluator()
        └─ GoEngine::genmove() → do_search() → HMGSDP_Optimized::search()
                │
                ▼
        src/go/search/hmgs_dp_optimized.cpp
                ├─ select_node_for_expansion()  UCT descent
                ├─ batch_eval_fn_()             batched leaf evaluation
                ├─ expand_node_with_prior()      top-K expansion + transposition merge
                └─ backpropagate()               value backup
                │
                ▼
        src/network/protocol/gtp_protocol.cpp
                └─ genmove → "= D4" and similar GTP responses
```

### Key files

| File | Role |
|------|------|
| [src/main.cpp](src/main.cpp) | Entry point, argument parsing, GTP / interactive / training dispatch |
| [src/decision/engine/go_engine.cpp](src/decision/engine/go_engine.cpp) | Engine facade, neural initialization, evaluator injection, genmove |
| [src/go/neural/ca_moe_expert.cpp](src/go/neural/ca_moe_expert.cpp) | CA-MOE forward pass, gating, policy/value fusion, save/load |
| [src/go/search/hmgs_dp_optimized.cpp](src/go/search/hmgs_dp_optimized.cpp) | HMGS-DP main loop, UCT, DAG merging, backpropagation |
| [src/perception/board/go_board_cuda.cpp](src/perception/board/go_board_cuda.cpp) | Bitboard operations, `encode_to_tensor()` feature encoding |
| [src/network/protocol/gtp_protocol.cpp](src/network/protocol/gtp_protocol.cpp) | GTP command implementation |

### About model files (important)

- **What is actually loaded:** the `*.pt` file passed to `--model=`, produced by the `spacego_train` binary.
- **Format:** LibTorch `torch::serialize::OutputArchive` — **not** TorchScript, and **not** a bare `state_dict`.
- **Required keys:** `backbone`, `gating_network`, `policy_head_0..7`, `value_head_0..7`, `num_experts`,
  `board_size`, `hidden_dim`, `is_training`.
- **Fixed structure:** 8 experts, 10 residual blocks, 17 input channels; `hidden_dim` is loaded as 256 with
  a fallback to 128.
- **Typical artifacts:**
  - `training_output_fresh/elo_gated/checkpoints/elo_best.pt` (best ELO-gated checkpoint)
  - `training_output/diploid/checkpoints/diploid_best.pt` (best diploid checkpoint)
  - `training_output_*/..._best.pt` (best unsupervised self-reinforcement checkpoints)
- **Not distributed with the repository:** a single checkpoint is ~237 MB, above GitHub's normal file
  limit. Train your own, or host a release checkpoint on Hugging Face / a GitHub Release and load it
  with `--model=`.
- **Without a model:** the engine falls back to rule-based search with heuristic evaluation and the
  neural network is not involved. This is not full playing strength.
- **Where to put models:** [models/README.txt](models/README.txt).

> The code contains **no** references to `models/hybrid_go_model.pt` or `models/ca_moe_expert.pt`.
> Those two filenames in an early README were leftovers and have been corrected.

---

## Core algorithms and innovations

### 1. HMGS-DP: hierarchical merged graph search with diversity protection

**H**ierarchical **M**erged **G**raph **S**earch with **D**iversity **P**rotection.

**(1) UCT and the parallel framework**

The selection rule is UCT-based. Nodes store visit counts, accumulated value, priors and virtual loss:

```text
UCT(i) = Q(i) + C · P(i) · √(ln N / n_i)
```

`HMGS_Node` already carries an atomic `virtual_loss` counter, and the repository provides a
`WorkStealingQueue` plus multi-worker configuration interfaces. However,
`HMGSDP_Optimized::search()` is currently a **single-threaded batched main loop**, and virtual loss is
not yet wired into selection or backup (see [known limitations](#known-limitations)). This layer is
scaffolding for later parallelism, not an active speedup.

**(2) Transposition-table state merging (tree → DAG)**

A transposition table keyed by Zobrist hash (stone layout + side to move):

```cpp
std::unordered_map<uint64_t, HMGS_Node*> tt_;
```

Equal positions reuse existing nodes through `add_child_ref`, turning the search structure from a tree
into a **directed acyclic graph**, so each position is evaluated once instead of repeatedly. The table
owns all nodes centrally, which avoids double-free when a DAG node is shared by multiple parents.

**(3) Batched inference path**

Once a batch evaluator is injected via `set_batch_evaluator()`, the search collects leaves according to
`batch_size` and evaluates them in a single forward pass, aiming to obtain policy priors and values from
one GPU call and to cut wall-clock time for self-play data generation. The current implementation has a
defect where the collection loop burns the iteration budget by `batch_size` and can select the same leaf
twice, so the effective simulation count is far below the configured value (P0, see
[known limitations](#known-limitations)).

**(4) Topology acceleration and persistence**

`SpatialTopologyAccelerator` computes spatial topology features in batches and shards work between CPU
threads and the GPU according to task size. The search tree, the merge network and the diversity graph
can be exported and imported (including v1 compatibility) for offline analysis.

> Design intent versus implementation status is documented in
> [docs/技术文档/算法说明.md](docs/技术文档/算法说明.md) (Chinese) and
> [known limitations](#known-limitations).

### 2. CA-MOE: causal-aware mixture-of-experts meta-decision

**C**ausal-**A**ware **M**ixture **o**f **E**xperts.

```text
input 17×19×19
   │
SharedBackbone: input conv + 10 × residual block + dropout (256 channels)
   │  shared features
   ├── ExpertPolicyHead × 8  → 362-dim policy per expert
   ├── ExpertValueHead  × 8  → 1-dim value per expert
   └── GatingNetwork
         fc1(hidden_dim×361 → 512) → fc2(512 → 8) → softmax
         + style_projection(5 → 8)  opponent-style bias

final_policy = Σ_e w_e · policy_e
final_value  = Σ_e w_e · value_e
w = softmax(Gating(features) + style_projection(opponent_style))
```

**What is new here:**

- **Shared trunk + multiple expert heads.** The trunk runs once; the 8 expert heads run in parallel,
  balancing capacity against inference cost.
- **Opponent-style conditioning.** A 5-dim style vector `[aggressive, defensive, balanced, territory,
  influence]` is added as a gating bias through a **zero-initialised** projection, which makes
  "choose experts based on the opponent's style" a learnable mechanism. Without an injected style the
  behaviour reduces to an ordinary gate, and older checkpoints still load (`load_compat`).
- **Preventing gate collapse.** During real training I hit the case where softening the temperature
  actually weakened regularisation. The final design fixes the softmax temperature at 1.0 and uses
  **KL / load-balancing regularisation + dropout + logits L2** instead. This failure is preserved in
  code comments and training logs.

### 3. Bitboards and feature encoding, written from scratch

- 19×19 = 361 points represented by `6 × uint64_t` bitmaps per color; placement, capture, liberties and
  connected components are all bit operations.
- Zobrist hashing mixes in a side-to-move component, used for the transposition table, experience
  deduplication and caching.
- `encode_to_tensor()` outputs a 17-channel interface; the current implementation fills the first 3
  planes (black, white, side to move) and leaves the rest at zero (training and inference agree on this
  convention). This is an extension point, see [known limitations](#known-limitations).

### 4. Training system (four modes)

| Mode | Entry point | Key mechanism |
|------|-------------|---------------|
| Supervised pre-training | `spacego_train --mode=supervised` | Synthetic data cold start, policy cross-entropy + value MSE |
| Unsupervised self-reinforcement | `spacego_train --mode=unsupervised` | Self-play + Dirichlet noise + replay buffer + progressive pool + AMP |
| Diploid competition | `spacego_train --mode=diploid` | Two model branches compete; accept/rollback by ELO threshold |
| ELO-gated online training | `spacego_train --mode=elo-gated` | Per-game learning + KataGo score gap as the yardstick + strict-improvement accept/rollback + teacher distillation |

**ELO gating** is the most distinctive part of the project: each round the current model plays and learns
against a frozen baseline, and the KataGo score gap is used as the yardstick. New parameters are
accepted only if the yardstick **strictly improves**, otherwise the model rolls back to the round-start
snapshot. It turns "did it actually get stronger" into an auditable per-round decision instead of
requiring you to trust a loss curve.

### 5. Engineering innovations

- **Evaluator injection decoupling.** The search depends only on `std::function` callbacks, so the neural
  evaluator, the rule-based evaluator and the batched evaluator are freely interchangeable.
- **Trimmable build.** `USE_CUDA` / `USE_LIBTORCH` / `BUILD_TRAIN` / `BUILD_GUI` / `BUILD_TESTS` are
  independent switches; the engine core builds without a GPU.
- **Graceful degradation.** Falls back to heuristic GPU utilisation when NVML is unavailable, to rule
  search when no model is loaded, and to per-node evaluation when batched evaluation fails.
- **Facade plus focused components.** The `HMGSDP` facade only handles lifecycle and forwarding; the
  search controller, UCT, similarity, merge network, node management and persistence are separate so
  they can be unit tested and replaced independently.

---

## Quick start

### Requirements

| Component | Version |
|-----------|---------|
| OS | Windows 10/11 (GUI is Windows-only); Linux can build the engine |
| Compiler | Visual Studio 2022 (MSVC v143) or GCC 11+ |
| CMake | 3.21+ |
| CUDA Toolkit | 12.x (optional, 12.4 recommended) |
| LibTorch | 2.6.0+cu124 (optional, required for neural inference and training) |

### Build

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DLIBTORCH_ROOT="D:/libtorch-win-shared-with-deps-2.6.0+cu124/libtorch"
cmake --build build --config Release
```

Artifacts land in `build/Release/`: `spacego.exe`, `spacego_train.exe`, `SpaceGo_GUI.exe`.

Minimal build (no CUDA / no LibTorch, rule engine only):

```powershell
cmake -S . -B build -DUSE_CUDA=OFF -DUSE_LIBTORCH=OFF -DBUILD_TRAIN=OFF -DBUILD_GUI=OFF
cmake --build build --config Release
```

> If the source path contains spaces or non-ASCII characters, MSBuild may split the arguments passed to
> `nvcc` at the spaces. Build from a pure-ASCII directory junction; see [BUILD_GUIDE.md](BUILD_GUIDE.md).

### Run

```powershell
# GTP mode (default): load a model and set simulations per move
.\spacego.exe --gtp --model=<checkpoint>.pt --simulations=1600

# Without a model: rule / heuristic fallback
.\spacego.exe --gtp --simulations=1600

# Interactive command line
.\spacego.exe --interactive

# Training
.\spacego_train.exe --mode=elo-gated --elo-rounds=300 --output-dir=./training_output
```

### Sabaki / Lizzie

- Engine path: `spacego.exe`
- Arguments: `--gtp` (or `--gtp --model=<path>` when a model is available)
- Working directory: the directory containing the executable

---

## Project structure

```text
include/
├── config/             platform and compile-time configuration
├── framework/          logging, performance monitoring, algorithm base classes, AMP gradient scaling
├── go/
│   ├── board/          bitboards, CUDA board, Zobrist hashing
│   ├── engine/         GoEngine wrapper
│   ├── heterogeneous/  CPU/GPU heterogeneous computing and load balancing
│   ├── neural/         CA-MOE expert network, CUDA inference, opponent-style profiling
│   ├── search/         HMGS-DP optimized implementation, CUDA search, topology acceleration
│   └── training/       the four training modes and tuning utilities
└── network/
    ├── algorithms/     HMGS-DP algorithm core (nodes / UCT / merging / persistence)
    └── protocol/       GTP protocol

src/                  implementations matching include/, including CUDA kernels
GUI/                  standalone Win32 + GDI+ graphical interface
tests/                GoogleTest unit tests (optional)
docs/                 technical documents and algorithm design notes (Chinese)
models/               model placement instructions (weights are not distributed)
```

> `network/algorithms` is a historical name: it holds the HMGS-DP search core and has nothing to do with
> network communication.

---

## Measured results and current status

The numbers below come from training logs and checkpoints inside this repository and are reproducible:

| Metric | Value | Notes |
|--------|-------|-------|
| CA-MOE parameters | ~62M | 8 experts, `hidden_dim=256` |
| Checkpoint size | ~237 MB | per `*.pt` |
| Simulations per move | 1600 (configurable) | ELO-gated training configuration |
| Time per move | ~35–55 s | GTX 1660 Ti, 1600 simulations |
| Score gap vs KataGo | −350 to −368 points | best observed **−349.5** (KataGo at 10 visits) |
| Code size | ~50k lines / 139 C++/CUDA files | including GUI and tests |

**Current positioning: research prototype.** The architecture, the training loop and the engineering
chain run end to end, but playing strength is far below mainstream engines, and the search and feature
encoding have the defects listed below. This project publishes those numbers rather than marketing an
unmeasured "industrial strength".

---

## Known limitations

> These come from a line-by-line review of the current code. Issues and PRs that fix them are welcome.
> Detailed analysis: [docs/技术文档/算法说明.md](docs/技术文档/算法说明.md) and
> [docs/技术文档/成果说明.md](docs/技术文档/成果说明.md) (Chinese).

| Level | Issue | Impact |
|-------|-------|--------|
| P0 | The batch-collection loop burns the iteration budget by `batch_size` and can select the same leaf twice within a batch | `--simulations=800` completes only ~6–7 effective expansions |
| P0 | UCT maximises child-perspective value without negating at the parent | Wrong selection direction, skewed search statistics |
| P0 | Value flip check uses `current_player == -1` while `StoneColor::WHITE == 2` | Values are not flipped when White is to move |
| P1 | `genmove` returning pass does not switch the side to move or count consecutive passes | Two consecutive passes cannot end the game properly |
| P1 | `boardsize` accepts 9/13, but coordinates, encoding and policy indices are hard-coded for 19×19 | Non-19×19 games misalign |
| P1 | CUDA board kernels are placeholders (liberties always 0, legality only checks occupancy) | GPU batched board computation is not usable in real play yet |
| P2 | Only the first 3 of 17 input planes are filled | Limited input expressiveness (consistent with the training convention) |
| P2 | GUI is not wired to the engine; `time_settings` / `stop` are unimplemented | The GUI runs standalone and does not play |

---

## Roadmap and contribution areas

**Near term (correctness first)**

1. Fix the three P0 defects — search budget, UCT value direction, White-side value flip — and add
   regression tests.
2. Unify the `StoneColor` / `±1` value conventions and add black/white symmetry tests.
3. Fix GTP pass and side-to-move synchronisation; implement `stop` and `time_left`.
4. Either declare 19×19 as the only supported size, or parameterise encoding and policy indices.

**Mid term (strength and performance)**

5. Extend the 17 input channels (history, liberties, ko, ladders) and retrain the baseline.
6. Upgrade the CUDA board kernels from placeholders to a real batched implementation and use them for
   self-play data generation.
7. Make batched inference the default on the GTP path and benchmark nodes/second and GPU utilisation.
8. Increase the number of evaluation games to reduce yardstick noise in ELO gating and to curb gate
   collapse.

**Long term (ecosystem)**

9. Publish downloadable pre-trained checkpoints and benchmark games.
10. Wire the GUI to the engine through a GTP subprocess.
11. Set up CI (Windows / Linux builds + unit tests + a GTP smoke test).

**Good first issues:** documentation fixes, GTP command completion, unit tests, benchmark scripts,
English documentation.

---

## Docs and license

- [Algorithm notes](docs/技术文档/算法说明.md)
- [Training log summary](docs/技术文档/训练日志总结.md)
- [Results notes](docs/技术文档/成果说明.md)
- [Build guide](BUILD_GUIDE.md)
- [Contributing](CONTRIBUTING.md)
- [Algorithm design notes](docs/算法数据/) (Chinese)
- [中文 README](README.md)

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.
If you modify it and run it as a network service, you must publish the modified source; derivative works
must also be licensed under AGPL-3.0. Full terms in [LICENSE](LICENSE).

---

<div align="center">

**SpaceGo** — a complete Go engine stack, opened up for you to read.

If this project helps you, a **Star**, a **Fork**, an Issue or a PR is genuinely useful.

</div>
