# Experimental Setup — Autobounded Neural Queries

Live planning note. Mirror of `lit/experimental_plan.md` and `lit/spelunking_table1_spec.md` in the paper repo (`~/papers/Autobounded-Neural-Queries`), kept here next to the implementation. Update this when the protocol changes.

## Algorithm targets (modules in this repo)

| Module | Algorithm | Spelunking analog | Headline win |
|---|---|---|---|
| `rayMarcher/` | Autobounded sphere tracing | `queries.py` ray casting | mid-pack vs. their `affine (fixed)` (honest) |
| `surfaceSampler/` | Autobounded uniform / blue-noise sampling via Ling et al. 2025 | naive rejection | works on non-SDF nets where sphere marching fails |
| `sphereWalker/` | Autobounded Walk-on-Spheres for grid-free PDEs | none | first guaranteed-WoS on neural fields |
| `meshExtractor/` | Adaptive MC + Dual Contouring + topology certification | `kd_tree.py` mesh extraction | beats `affine (full)` in 3D speed + memory |

## Benchmark shape set (10 shapes)

| # | Shape | Category | Source | Status |
|---|---|---|---|---|
| 1 | bunny | scan / character (Stanford) | Spelunking `sample_inputs/bunny.npz` | matched |
| 2 | fox | character | Spelunking `sample_inputs/fox.npz` | matched |
| 3 | hammer | mechanical | Spelunking `sample_inputs/hammer.npz` | matched |
| 4 | birdcage_occ | mechanical, **occupancy** (non-SDF) | Spelunking `sample_inputs/birdcage_occ.npz` | matched |
| 5 | jeb | mechanical / TO benchmark | downloaded `.obj` | augmentation |
| 6 | rocker-arm | mechanical (FE classic) | downloaded `.obj` | augmentation |
| 7 | armadillo | character / scan (Stanford) | downloaded `.obj` | augmentation |
| 8 | dragon | scan (Stanford) | downloaded `.obj` | augmentation |
| 9 | happybuddha | scan (Stanford, high genus) | downloaded `.obj` | augmentation |
| 10 | bob | character (Crane) | downloaded `.obj` | augmentation |

Composition: 4 mechanical / 3 characters / 3 scans, matching Spelunking's stated category mix.

**Reproducibility framing:** 4 of 10 shapes match Spelunking 1:1 (run their released code on these for direct comparison); 6 are augmentation drawn from public sources to cover the same category mix. Document explicitly in the paper.

**Birdcage_occ is the explicit non-SDF demo** — anchor the "we work on arbitrary neural fields, not just SDFs" pitch on it.

## Architecture sweep (NEW vs. Spelunking)

Spelunking fixes the architecture at 7 layers × 32 width, ReLU+ELU. We sweep both depth and width to characterize how bound tightness scales — this is one of our distinguishing ablations.

**Default sweep grid:**

| Knob | Values |
|---|---|
| Hidden layers | 3, 5, 7, 9, 11 |
| Hidden width | 32, 64, 128, 256 |
| Activation | ReLU, ELU, SoftPlus, sin (SIREN-style) |
| Loss | SDF (L1), occupancy (cross-entropy) |

This is ~5 × 4 × 4 × 2 = 160 configurations. Full grid is overkill; pin a reasonable subset:

- **Primary sweep (matches Spelunking and extends it):** depth ∈ {3, 5, 7, 9}, width ∈ {32, 64, 128}, activation ∈ {ReLU, ELU}, loss ∈ {SDF, occupancy} → 4 × 3 × 2 × 2 = 48 nets per shape × 10 shapes = 480 nets. Tractable on a single GPU node over a couple of days.
- **Activation-stress sweep:** at fixed depth=7, width=64, train SoftPlus and sin variants on the four "easy" shapes (bunny, fox, bob, armadillo) → 4 × 2 = 8 extra nets. Tests autobound's behavior on smoother (SoftPlus) and high-frequency (sin) activations.
- **Network-size scaling (probes affine-truncate territory):** at activation=ReLU, loss=SDF, push to depth=11, width=256 on bunny + jeb + happybuddha → 3 extra nets. ~2800 nonlinearities; this is where Spelunking's `affine (truncate)` is supposed to win, so we want this regime in the comparison.

Total: roughly **490 trained networks**.

**Why this sweep matters for the paper:**

- Spelunking's bound-machinery comparison (Table 1) is a single-architecture story. We extend to a 2-D (depth × width) grid → two-axis figures replacing their single-row table. This is a structural strength of the Taylor-model framing: the polynomial cost is well-defined per architecture, and we can characterize the cost-vs-tightness tradeoff as architectures scale.
- Probes the affine-truncate-wins-at-scale claim head-on. If autobound deg-2 wins on 3D queries at 7×32 *and* at 11×256, the headline story is robust.
- Provides the data for a "which bound machinery should you use, given your architecture?" recommendation table — practical reviewer-facing deliverable.

## Hyperparameters pinned from Spelunking's released code

(Source: `~/neural-implicit-queries`; full spec in `~/papers/Autobounded-Neural-Queries/lit/spelunking_table1_spec.md`.)

| Param | Value | Notes |
|---|---|---|
| `data_bound` | 1.0 | unit cube/sphere normalization |
| `box_width` | 0.05 | half-extent of random benchmark boxes |
| Region sampling | axis-aligned, uniform centers | not arbitrarily oriented |
| `n_keep` (truncate) | 8 | `benchmark_bounds.py:180` |
| `n_append` | 4 | `benchmark_bounds.py:181` |
| `truncate_policy` | `absolute` | sort by absolute magnitude |
| `affine (fixed)` rule | `delta` folds directly into `err`, no new affine column | `affine.py:175` |
| Slope-interval rule | primal + per-coord slope envelope | `slope_interval.py:30-50` |
| Raycast camera | `eye=[2.5,2.0,2.5]`, `target=[0,0,0]`, `up=y`, `fov=30°` | `main_render_headless.py:22-26` |
| Raycast resolution | 1024 × 1024 | confirmed by user; matches `main_render_headless.py:22` default |
| Float precision | fp32 (JAX default) | implied |

**Training is not part of the comparison.** Our contribution is purely about evaluating pre-saved neural fields. Whatever pipeline produced a network, the same network is queried by all bound variants — that is the apples-to-apples comparison. Use Spelunking's shipped networks as-is for the 4 matched shapes; for the 6 augmentation shapes, train networks ourselves with whatever pipeline is convenient (need not match Spelunking's training hyperparameters), since training quality is upstream of bound-machinery performance.

## Bound-machinery variants compared

| Label | Source | Status |
|---|---|---|
| `interval` | `~/neural-implicit-queries/src/affine.py` mode `interval` | reproduce verbatim |
| `affine (full)` | mode `affine_all` | reproduce verbatim |
| `affine (fixed)` | mode `affine_fixed` | reproduce verbatim |
| `affine (truncate, n=8)` | mode `affine_truncate`, `n_keep=8` | reproduce verbatim |
| `affine (append, n=4)` | mode `affine_append`, `n_append=4` | reproduce verbatim |
| `slope-interval` | `slope_interval.py`, Ratz 1996 | port verbatim |
| `autobound deg-1` | ours | implement |
| `autobound deg-2` | ours | implement (default for DC, surface sampling, WoS) |
| `autobound deg-3` | ours, scalability/curvature-stability ablation | implement |

Drop `affine (append)` from the headline comparison if it remains uncompetitive — Spelunking themselves note no advantage vs. truncate or fixed.

## Tables / figures to produce

1. **Spelunking Table 1 reproduction + extension.** All 9 bound variants × {1D time, 1D conclusive length, 3D time, 3D conclusive volume, raycast time}, on the architecture sweep. Output: 5-axis grid (variant × shape × depth × width × activation), aggregated to summary tables in the paper and full grid in the appendix.

2. **Spelunking Figure 6 reproduction.** Image-error vs. render-time on a 256×256 view. All variants. Honest mid-pack on rays.

3. **Empty-sphere throughput** (Spelunking Sec. 4.4). Headline 3D win.

4. **Closest-point latency, hierarchical mesh extraction wall-clock, bulk properties** — secondary 3D wins.

5. **Mesh extraction tiered showcase** (Tier 1: adaptive MC; Tier 2: DC with Hermite normals; Tier 3: sharp-feature DC with curvature; Tier 4: topology certificate). Each on the relevant subset of shapes.

6. **Surface sampling via Ling et al. 2025** with autobound replacing sphere marching. Headline regime: non-SDF networks (`birdcage_occ` plus our occupancy-trained augmentation shapes).

7. **Autobounded Walk-on-Spheres** for grid-free PDE solves on neural-field boundaries. Headline regime: TO-style problems on `jeb` / `rocker-arm` (ties to `radler2025diverse`).

8. **Architecture-scaling figure (NEW vs. Spelunking).** For each bound variant, plot 1D / 3D conclusive-region size as a function of architecture parameter count (depth × width). Reveals where each variant's regime of competitiveness lies. Practical, reviewer-facing deliverable.

## Hardware

Document the GPU we run on (different from RTX 2070). Report **both** absolute timings and ratios relative to `affine (full)` on the same hardware. The ratios are the apples-to-apples claim; absolute timings are auxiliary.

## What needs Spelunking's code in the comparison loop

For the 4 matched shapes: run their `benchmark_bounds.py` with our chosen `--batch_sizes` to produce paired numbers on identical hardware. Pin `n_truncate=8`, `n_append=4`, `truncate_policy=absolute` (their defaults). For augmentation shapes: train our own networks at the matched architecture and report their kernels' timing alongside.

## Decisions locked in (2026-04-29)

- **Raycast resolution: 1024×1024.** Matches `main_render_headless.py:22` default.
- **Hash-grid encodings: OUT of scope.** Joint limitation of the field — no bound machinery (interval, affine, autobound, slope-interval) handles non-smooth input encodings effectively. Document in the paper as a category-level limitation, not a method-specific weakness. Hash-grid networks are not part of the architecture sweep.
- **Training not part of the comparison.** This work evaluates pre-saved networks. Use Spelunking's shipped networks for the 4 matched shapes; train our own (any reasonable pipeline) for the 6 augmentation shapes. Training quality is upstream of the bound-machinery comparison.
- **Reporting format = per-architecture rows.** Tables and figures break out results by architecture (depth × width × activation), no cross-architecture aggregation. This produces an architecture-scaling figure as a load-bearing deliverable, replacing Spelunking's single-architecture Table 1 with a 2-D characterization grid.
- **Timing claim format = ratios on same hardware.** Always report (our deg-2 time) / (`affine (full)` time) on identical hardware, alongside absolute times. Hardware-independent ratios are the apples-to-apples claim.
