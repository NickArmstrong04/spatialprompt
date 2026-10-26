# CLAUDE.md — meta_spatialomics

Text-promptable cell segmentation **and** phenotyping for spatial omics imaging, in one
step. A user types a plain-English query ("CD8+ T cells at the tumor boundary") and gets
back the masks for exactly those cells plus per-cell transcript or protein counts.
Built on Meta's SAM 3.1 (text-prompted segmentation), Muse Spark (agent that repairs
segmentation errors), and Muse Code (the coding agent used to build the pipeline).

Keep this file updated as decisions are made. It is the context Claude (and Muse Code)
reads first.

## Project status

- Started **2026-10-20**. Ten-day build window: target end-to-end demo by **2026-10-30**.
- Day 1 done: data on disk, 5 ROIs extracted, Cellpose baseline scored (see Baseline).
  Public repo: https://github.com/NickArmstrong04/spatialprompt (created 2026-10-26, MIT).
  Commit `b37d4d5`, tag `pre-hackathon`, holds all Oct 20–21 work with its real author
  date; everything after it is hackathon work. Keep that boundary honest: never backdate
  or re-date commits. `data/`, `vendor/`, `.env` are gitignored.
- Owner: Nick, computational biologist. The demo must run on data he actually works with.
- Local machine: Windows 11, Python 3.13.5 (miniconda base), torch 2.9.1 + torchvision
  0.24.1 (CUDA build not yet confirmed), NVIDIA RTX 4070 Ti SUPER 16 GB, git 2.50.
  spatialdata 0.8.0 + zarr 3.4.1 installed in base on 2026-10-20 (pandas 3 now conflicts
  with streamlit there; irrelevant for us). Cellpose, scanpy, tifffile not installed.
  Muse Code **not** installed. Console is cp1252: run scripts with `python -X utf8`.
- Event: **Meta Global AI Developer Hackathon** (Cerebral Valley). Submissions due
  **Wed 2026-11-04, 9:00 PM ET**. See the Hackathon section below. The deadline is the
  real end date; the "ten-day" Oct 30 target is now an internal milestone.

## Hackathon: rules, judging, deadlines (recorded 2026-10-26 from the participant guide)

Source: https://cerebralvalley.ai/e/meta-ai-developer-hackathon/details (JS-rendered;
WebFetch sees only "Loading...", read it in a browser). All times **Eastern**.

**Problem statements (the project must build toward one).** We fit **"Push the
Frontier"**: use Meta's models to "accelerate a scientific experiment" / give a machine a
new capability. Name it explicitly in the submission. (The other one, "Build for Your
World", is also defensible because this is Nick's own domain.)

**Models that count:** Muse Spark series, Muse Wonder, Muse Image, Muse Voice Transcribe
and **SAM 3.1 through the Meta Model API**; open-weight **Muse Glimmer** and **Muse Spark
Open**; **Muse Code** coding agent. Stage-one judging is pass/fail on "reasonably used
Meta Model API and/or Meta's open-weight models". So Muse Spark on the API must actually
run in the demo (it is still untested here, see Query parser), and Muse Code use must be
real and shown (keep the session JSONLs).

**Schedule**
| When (ET) | What |
|---|---|
| Mon Oct 26, 12:00 PM | Kickoff (rules, prizes, judging, Meta speakers), livestream |
| Tue Oct 27, 12–1 PM | Agents 101: Build a Multimodal AI Agent (Suraj Subramanian) |
| Wed Oct 28, 12–1 PM | Agents 102: Best Practices for Verifiable Agents (Yash Khare) |
| Fri Oct 30, 12–1 PM | Specialized Intelligence on Fireworks with Muse Open Models (Sinan Ozdemir) |
| Weekdays, 10–11 AM | Text office hours in Discord `#office-hours` (Meta team) |
| Oct 27 – Nov 4 | **Official building period** |
| Tue Nov 3 | Last full day for submission questions / troubleshooting |
| **Wed Nov 4, 9:00 PM** | **Submissions due** on the CV platform; also last moment to claim API credits |
| Thu Nov 5 – Sun Nov 8 | Judging; Most Viral engagement counted until Nov 8 EOD |
| Mon Nov 9, 12:00 PM | Closing: top 10 (5 per track), live final round, winners, community vote opens |
| Mon Nov 16 | Community-vote winner announced |

**Rules**
- Repository must be **public** (open source). One submission per team, entered by one
  "Representative"; everyone who worked on it listed. Late submissions not considered.
- Team size ≤ 4; solo allowed.
- **New work only.** What is presented must have been built during the event. The demo
  must show only features/code built during the hackathon, and judges must be able to tell
  what was created then; failing that is **disqualification at the judges' discretion**.
  Startups may build on existing code but must mark what is new.
  ⚠ This project started 2026-10-20, before the Oct 26 kickoff / Oct 27 building period.
  Unresolved: decide how to handle the pre-event work (Nick's call). At minimum, the repo
  history and README must say honestly what existed before Oct 26 and what was built after.
- Disqualified if the project violates legal/ethical/platform policies or uses code, data
  or assets without rights. Check: Xenium is CC BY 4.0 (attribute); **TissueNet is
  noncommercial-licence**, never commit it to the public repo; SAM License for SAM 3.
- **Anti-projects to stay clear of:** AI that generates medical advice; basic RAG;
  "simple image analyzers"; simple education chatbots; job screeners; nutrition coaches;
  personality analyzers; sports coaches; political/war. Frame this as a research tool for
  spatial-omics analysis (no diagnosis, no clinical advice), and make the depth visible
  (agentic QC loop, benchmarks) so it does not read as a "simple image analyzer".

**Submission:** public repo URL, **3-minute demo video** (a primary judging artifact
together with the code and the live product) showing only hackathon-built features.
Double-check the repo and video links are publicly accessible.

**Judging**
1. Stage one, pass/fail: fits a problem statement and reasonably uses Meta Model API
   and/or Meta open-weight models.
2. Stage two, each criterion scored 0–10:

| Criterion | Weight | What they ask |
|---|---|---|
| Impact | 30 % | Real-world potential, who benefits, could people actually use it, fits a problem statement |
| Use of Meta AI tooling | 25 % | Creative use, beyond basic integration, capabilities that "surprised even us" |
| Demo | 25 % | Working, impressive, holds up live, video is cool to watch |
| Technical depth & execution | 20 % | Sound, refined engineering, built with craft |

3. Stage three: top 5 per track; final judges watch the videos live and rank with the same rubric.

**Prizes** ($1M total). Indie Developer track $400K (150/100/75/50/25K); Startups track
$550K (250/150/75/50/25K). Bonus prizes ($50K):
- Most Viral Project $20K: most X engagement on the demo post through Nov 8 EOD; submit the post link.
- Best Build on Meta Open-Weight Models $10K: Muse Glimmer or Muse Spark Open on own
  hardware or partner inference (verified by the partner).
- Community Vote $10K: public gallery, one vote per verified account.
- **Best Agentic Use-Case $5K**: autonomy, task effectiveness, usefulness. Our QC agent
  (detect → plan → re-prompt SAM → re-detect) is the natural fit; a working MusePlanner matters for this.
- Best Creative Build $5K: Muse Image / Muse Wonder (not us).

**Credits and resources**
- **Meta Model API: $150 per participant**, claimed individually via the Google form
  linked in the guide, by Nov 4 9 PM ET. Arrive in batches; check the Billing page. A
  credit card must be on the account (temporary pre-auth hold). Set a Spend Limit alert.
- Baseten: request $100 + 2M TPM (up to 2,000 people, reviewed by use case, per person).
  **Use model id `meta-models/muse-spark-open-1026`**, not the docs' `meta-models/muse-spark-open`.
- Modal: $100 to the first 1,000 participants, for Meta open-weight models via Endpoints.
- Fireworks: $25 per team, shortlist only. Partner credits are not guaranteed.
- Meta docs/cookbooks listed in the guide: Overview, Quickstart, Model list, API cookbook,
  API fundamentals, Muse Code recipes, OSS model cookbook; on-demand talks on Muse Spark.
- Discord: `#announcements`, `#questions` (Cerebral Valley), `#office-hours` (Meta),
  `#team-search`, Baseten channel.

## The pitch (one paragraph, keep verbatim for submissions)

> A text-promptable cell segmentation and phenotyping tool for spatial omics imaging.
> Current pipelines (Cellpose, Mesmer, 10x's default) draw cell boundaries from nuclear
> and membrane stains and leave phenotyping to a separate clustering step. I want to
> collapse the two: a user types a plain-English query such as "CD8+ T cells at the tumor
> boundary," and the tool returns the masks for exactly those cells with their per-cell
> transcript or protein counts. The pipeline composites the relevant marker channels into
> an RGB image, uses SAM 3.1's text-prompted segmentation to produce phenotype-specific
> masks, and assigns transcripts or protein intensities to each mask. Muse Spark runs as
> an agent over the output, flagging doublets and split cells and re-prompting SAM 3.1 to
> repair them. Muse Code is used to build the pipeline. I will demo on a public 10x Xenium
> or CODEX dataset, restricted to a handful of regions of interest so the full loop runs
> end to end within the ten days. Segmentation quality will be benchmarked against
> Cellpose on TissueNet (F1 at IoU 0.5), and phenotype assignment against marker-based
> labels on the Xenium slide. The result is segmentation and phenotyping in one step, in
> plain language, on data I work with in computational biology.

## Pipeline design

```
query ("CD8+ T cells at the tumor boundary")
  │
  ├─ 1. Query parser (Muse Spark)  → {markers: [CD8A, CD3E], context: "tumor boundary",
  │                                   negative_markers: [...], sam_prompt: "cell"}
  ├─ 2. Channel compositor         → RGB image per ROI: e.g. R = marker-of-interest,
  │                                   G = membrane/boundary stain, B = DAPI
  ├─ 3. SAM 3.1 text-prompted seg  → masks, boxes, scores for the noun-phrase prompt
  ├─ 4. Quantifier                 → per-mask transcript counts (Xenium) or mean
  │                                   protein intensity per channel (CODEX)
  ├─ 5. Phenotype gate             → keep masks whose counts/intensities satisfy the
  │                                   parsed marker logic; spatial filter for "boundary"
  ├─ 6. QC agent (Muse Spark)      → flags doublets (two nuclei in one mask), split cells
  │                                   (one nucleus across two masks), empty masks; issues
  │                                   repair actions = re-prompt SAM 3.1 on a crop with a
  │                                   box/point prompt, or merge/split masks
  └─ 7. Output                     → label image (uint32 TIFF), masks (GeoJSON/parquet),
                                      per-cell table (cell_id, x, y, area, counts...),
                                      overlay PNG for the demo
```

Design rules:
- **SAM 3.1 takes short noun phrases only.** It does not understand "CD8+ T cells" or
  referring expressions. The prompt to SAM is always a simple visual noun phrase
  ("cell", "nucleus", "bright red cell", "round cell"). The biology lives in the channel
  compositing (step 2) and the marker gate (step 5), not in the SAM prompt. Muse Spark
  does the translation from biology to those two things.
- **Work in ROIs, not whole slides.** A handful of 1024–2048 px crops per dataset. Whole-
  slide is out of scope for the ten days.
- **Every stage is a plain function with a file-based input/output** so Muse Code can
  build and test them independently and the QC agent can call them as tools.
- **Keep the Cellpose baseline runnable** in the same harness; the benchmark is only
  credible if both models see identical crops and identical ground truth.

## Models and APIs (verified 2026-10-20)

### SAM 3.1
- **Verified from the code (vendor/sam3, commit 7f473cf, 2026-10-14): SAM 3.1 is a video
  checkpoint.** `facebook/sam3.1` contains only `sam3.1_multiplex.pt`, loaded by
  `build_sam3_predictor(version="sam3.1")` (video, Object Multiplex). The **image** model
  `build_sam3_image_model()` always loads `sam3.pt` from `facebook/sam3`; there is no
  3.1 image checkpoint. For our still-image pipeline "SAM 3.1" therefore means the SAM 3
  image model (text + box prompts via `Sam3Processor.add_geometric_prompt`), and the
  3.1 multiplex predictor only if we ever treat z-stacks or ROI tiles as a video.
  Wording for the write-up: "SAM 3 / 3.1 (Segment Anything with Concepts)". Request
  access to **both** HF repos; `hf auth login` once (token needs read access).
- Image processor resizes any input to 1008 × 1008, so feed ≤ 1024 px tiles (a 2048 ROI
  = four tiles) or cells shrink to ~12 px. `confidence_threshold` default 0.5; lower
  it for dense small objects. Box prompt format: [cx, cy, w, h] normalised to [0, 1].
- Installed into the `spatial` env with `pip install -e vendor/sam3` plus fixes:
  `einops`, `pycocotools`, `triton-windows` (edt.py imports triton at module load), and
  numpy forced back to 2.4 (sam3 pins `numpy<2` but tifffile/zarr need ≥ 2.1; works).
  `scripts/sam3_smoke.py` is the first thing to run once the HF token is in place.
- License: SAM License (see repo).
- Code: https://github.com/facebookresearch/sam3 (updated for 3.1, see `RELEASE_SAM3p1.md`).
  Requires Python 3.12+, PyTorch 2.7+, CUDA 12.6+. Install: clone, `pip install -e .`.
  Use a dedicated Python 3.12 env (conda or uv) to match the README; confirm the torch
  build is CUDA, not CPU.
- Image API (from README):
  ```python
  from sam3.model_builder import build_sam3_image_model
  from sam3.model.sam3_image_processor import Sam3Processor
  model = build_sam3_image_model()          # check RELEASE_SAM3p1.md for the 3.1 ckpt arg
  processor = Sam3Processor(model)
  state = processor.set_image(pil_image)
  out = processor.set_text_prompt(state=state, prompt="cell")
  masks, boxes, scores = out["masks"], out["boxes"], out["scores"]
  ```
  Box prompts are covered in the image notebook; point refinement in the video notebook.
- 3.1 vs 3: drop-in replacement; object multiplexing (16 objects per forward pass) and
  global reasoning, ~2x faster video, lower GPU requirements. Text prompting unchanged.
- Hosted alternative: Meta Model API, model ID `sam-3-1`, text prompts only, OpenAI-SDK-
  compatible client, $2.50 per 1k images. Docs: https://dev.meta.ai/docs/media-segmentation/
  Prefer local weights (we need box/point re-prompting for repair); hosted is the fallback
  if the local install fights us.

### Muse Spark (the QC / query-parsing agent)
- Meta Model API, self-serve, OpenAI-SDK-compatible. Model IDs seen: `muse-spark-1.2`,
  `muse-spark-1.2-contributor`; 1.3 announced as available in Muse Code and the API.
  1M-token context. Also on OpenRouter (`meta/muse-spark-1.2`).
- Pricing (1.2 standard): $1.25/M input, $0.15/M cached input, $4.25/M output; reasoning
  tokens billed as output. Contributor tier is cheaper ($0.10/$0.20 for 1.3) but requests
  may be used for training.
- Supports tool calls and parallel tool calls. Vision/image input: **not confirmed**. If
  it cannot see images, the QC agent works from numeric features per mask (nucleus count,
  area, solidity, overlap with neighbors, marker intensity). Confirm on day 1.
- Docs: https://dev.meta.ai/docs/ . Blog: https://dev.meta.ai/resources/blog/build-with-muse-spark/

### Muse Code (the coding agent)
- Install: `curl -fsSL https://dev.meta.ai/install.sh | bash` (page shows a Windows tab
  without a command; check it, or install under WSL/Git Bash). Auth in browser at
  dev.meta.ai. Run `muse` in the project dir; `muse resume`; `/model`, `/effort`.
- Sessions are JSONL under `~/.local/share/muse/sessions/`. Keep these: they are evidence
  that Muse Code built the pipeline, which matters for the write-up.
- Blog: https://dev.meta.ai/resources/blog/build-with-muse-code/

## Data

### Demo dataset: DECIDED 2026-10-20 → 10x Xenium Prime 5K, FFPE human breast cancer

**Primary: Xenium Prime 5K FFPE Human Breast Cancer** (XOA 3.0.0, CC BY 4.0).
https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer
- Panel: 5K Pan Tissue & Pathways + 100 custom = 5,101 genes. Verified present:
  CD8A, CD8B, CD3E, CD3D, CD4, FOXP3, MS4A1, CD19, CD68, GZMB, PDCD1, EPCAM, ERBB2, ESR1,
  MKI67, PECAM1. Verified **absent** (by name in gene_panel.json): PTPRC, KRT8/18/19,
  ACTA2, VIM, COL1A1, NKG7. So "T cell" = CD3E/CD3D, "tumor" = EPCAM/ERBB2/ESR1,
  "stroma" = use the αSMA/Vimentin **stain** channel, not transcripts.
- Multimodal segmentation stains, `morphology_focus/` channel order (verified from the
  OME metadata): 0 = DAPI, 1 = ATP1A1/CD45/E-Cadherin (boundary), 2 = 18S (interior
  RNA), 3 = AlphaSMA/Vimentin (interior protein). Native pixel size 0.2125 µm.
- 10x's own segmentation on the slide: 72 % by interior stain, 26 % by boundary stain,
  2 % nucleus expansion (from the crop below). ~407k cells on the whole slide.
- Full bundle `Xenium_Prime_Breast_Cancer_FFPE_outs.zip` is **41 GB**; do NOT download it
  whole. The CDN (cf.10xgenomics.com) honours HTTP Range requests, and the zip's central
  directory lists members, so pull only what we need by member (all deflate-compressed,
  so each member streams in full): the four `morphology_focus_000{0..3}.ome.tif`
  (2.1 / 2.7 / 1.7 / 2.2 GB), `transcripts.parquet` (1.7 GB), `cells.parquet`,
  `cell_boundaries.parquet`, `nucleus_boundaries.parquet`, `cell_feature_matrix.h5`
  (0.09 GB), `gene_panel.json`. ≈ 11 GB total. Skip `morphology.ome.tif` (24.6 GB,
  z-stack). H&E whole-slide is a separate 6 GB download
  (`..._he_image.ome.tif`) — optional, for figures only.
  Base URL: `https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Breast_Cancer_FFPE/`
- **Already on disk**: `data/raw/zenodo_bc_crop/` = EMBL's SpatialData crop of this slide
  (Zenodo 20180722, CC BY 4.0): 18,626 cells, 5,101 genes, 2.98 M transcripts, cell +
  nucleus polygons, region x 12,360–19,564 µm, y 39,962–48,509 µm (7.2 × 8.5 mm).
  Its images are downsampled to **4 µm/px** (2111 × 1787) so they are useless for
  segmentation, but the tables, polygons, and transcripts are full-fidelity: use it to
  pick ROIs and as phenotype ground truth on day 1–2 before the big image members arrive.
  In this crop CD8 T cells are sparse (184 cells with CD8A ≥ 2, 29 with CD8A ≥ 2 & CD3E ≥ 1;
  MS4A1 ≥ 2: 307; ERBB2 ≥ 2: 384; ESR1 ≥ 2: 437). Pick immune-rich ROIs from the
  whole-slide `cells.parquet` + `cell_feature_matrix.h5` instead of this crop, and have
  a second demo query with an abundant phenotype ("HER2+ tumor cells", "B cells").
- `data/raw/panels/breast_prime_panel.json` and `lung_mm_panel.json` are the gene panels.

**Fallback / second slide: FFPE Human Lung Cancer with Xenium Multimodal Cell
Segmentation** (XOA 2.0.0, Multi-Tissue & Cancer Panel, 377 genes + add-on = 397
targets, 162k cells, median 46 transcripts/cell). Whole bundle is only **8 GB**
(`https://cf.10xgenomics.com/samples/xenium/2.0.0/Xenium_V1_humanLung_Cancer_FFPE/Xenium_V1_humanLung_Cancer_FFPE_outs.zip`),
morphology_focus channels 0.4–0.5 GB each. Panel has CD8A, CD3E, CD3D, CD4, FOXP3,
PTPRC, NKG7, GZMB, MS4A1, CD68, EPCAM, ACTA2 (no CD8B). Same stain kit, so the pipeline
should transfer unchanged; use it on day 8–9 if there is time to show generalization,
or switch to it on day 2 if the breast image download fights us.

Rejected: Janesick 2023 breast (XOA 1.0.1, DAPI only, 313 genes, no boundary stain);
tonsil/other Multi-Tissue previews (DAPI only); Xenium Prime lung (22.6 GB, 5K panel,
no advantage over breast); CODEX (Schürch CRC on TCIA is 2 TB; HuBMAP needs its own
reader; no transcripts; drop unless the Xenium image path fails).

- Xenium has no CD8 **protein** channel; "CD8+" is gated on `CD8A`/`CD8B` transcripts
  within the mask, with `CD3E` as co-marker. The RGB composite for immune queries uses a
  transcript-density raster for the marker channel (KDE of transcript points at pixel
  resolution), the boundary stain, and DAPI.

### Benchmark dataset
- **TissueNet** (Greenwald et al. 2022, Nat Biotech, Mesmer paper): >1M paired whole-cell
  and nuclear annotations, nine organs, six platforms. Download from
  https://datasets.deepcell.org/ (noncommercial, registration required). Use the test split.
- **On disk** (2026-10-20): `data/tissuenet/tissuenet_v1.1.zip` (4.4 GB; train 2.87 GB,
  val 1.22, test 0.37) with the test split extracted. Verified layout: `X` (n, 256, 256,
  2) float64 raw counts [nuclear, whole-cell], `y` (n, 256, 256, 2) int32 [whole-cell
  labels, nuclear labels], `meta` (n+4, 6) object with 4 header rows first then one row
  per image (filename, experiment, pixel_size, screening_passed, time_step, specimen).
  Test = 1,324 tiles (331 originals × 4 quadrants): GI 396, pancreas 308, breast 196,
  tonsil 168, lung 152, lymph node 52, epidermis 48, spleen 4; pixel size 0.5 µm for
  1,080 tiles, else 0.365–1.0. Roughly 200–350 cells per tile. Noncommercial licence.
- Metric: **F1 at IoU 0.5**, matched instances, same as the Mesmer paper. Baseline:
  Cellpose `cpsam` on the same crops with the same 2-channel input, via
  `bench/tissuenet_cellpose.py` (run `--inspect` first on a fresh download to confirm
  the npz keys and channel order). Expected location: `data/tissuenet/` (zip or npz).
  Report per-platform if time allows.
- Phenotype benchmark: on the Xenium ROIs, marker-based labels from 10x's own cells +
  cell_feature_matrix (threshold CD8A/CD3E per cell) vs. our query output; precision/
  recall per query.

## Baseline (done 2026-10-20, day 1)

What exists and how to regenerate it. All commands use the `spatial` conda env
(`C:/Users/nicka/miniconda3/envs/spatial/python.exe`, Python 3.12, torch 2.11+cu128,
cellpose 4.2.1, spatialdata 0.8). `conda run` cannot take multi-line `-c` scripts; call
the env's python directly.

1. `scripts/fetch_xenium.py` — selective download of bundle members over HTTP Range
   (handles the CDN's occasional 200-instead-of-206). `data/raw/xenium_breast_prime/`
   now holds the 4 morphology_focus OME-TIFFs, transcripts.parquet, cells / boundaries
   parquet, cell_feature_matrix.h5, gene_panel.json, experiment.xenium (10.5 GB, ~7 min).
2. `scripts/pick_rois.py` — scores 100 µm-step windows of 2048 px (435 µm) by CD8 T-cell
   count × tumor/non-tumor mix. Rules (5K panel counts are sparse, CD8A ≥ 1 in only 3 % of
   cells): CD8 T = CD8A+CD8B ≥ 1 & CD3E+CD3D ≥ 1 (4,651 on slide); tumor =
   EPCAM+CDH1+ERBB2+ESR1+GATA3+FOXA1 ≥ 3 (70,873). Output `data/roi/rois.json` +
   `roi_overview.png`. Five ROIs chosen, 2,000–2,160 cells and 40–95 CD8 T each.
3. `scripts/extract_rois.py` — per ROI: `image.tif` (4, 2048, 2048) uint16,
   `composite.png` (R = boundary stain, G = 18S, B = DAPI), 10x masks rasterized from
   the boundary polygons as `cells_10x.tif` / `nuclei_10x.tif` (uint32, ids = 10x
   label_id), `cells_10x.parquet`, `counts_10x.npz` + `counts_10x_genes.npy` (5,101
   genes × cells, sparse), `transcripts.parquet` (with x_px/y_px), `roi.json`.
   Mask-to-image alignment checked visually: 10x polygons sit on the cells.
   Coordinate convention: px = µm / 0.2125, origin = morphology image (0, 0).
4. `bench/cellpose_baseline.py` — Cellpose `cpsam` on [boundary stain, DAPI] for cells
   and DAPI alone for nuclei; matched-instance F1 at IoU 0.5 against 10x's masks.
   Writes `cells_cellpose.tif`, `nuclei_cellpose.tif`, `overlay_cellpose.png` (cyan =
   10x, yellow = Cellpose) per ROI and `results/baseline_cellpose.csv`.

| ROI | 10x cells | Cellpose cells | cell F1@0.5 (P / R) | nucleus F1@0.5 |
|---|---|---|---|---|
| roi01 | 2442 | 2212 | 0.824 (0.87 / 0.79) | 0.718 |
| roi02 | 2463 | 2186 | 0.804 (0.86 / 0.76) | 0.726 |
| roi03 | 2407 | 2135 | 0.790 (0.84 / 0.75) | 0.701 |
| roi04 | 2423 | 2102 | 0.781 (0.84 / 0.73) | 0.717 |
| roi05 | 2330 | 2042 | 0.713 (0.76 / 0.67) | 0.667 |
| mean | | | **0.782** | **0.706** |

**TissueNet test, human labels** (`bench/tissuenet_cellpose.py --split test --n 0`,
`results/tissuenet_cellpose_test.csv`, 1,324 tiles, ~4.5 min on the GPU):
Cellpose cpsam whole-cell **F1@0.5 = 0.854** (P 0.899, R 0.817). Per tissue: tonsil
0.918, lung 0.907, lymph node 0.894, breast 0.862, pancreas 0.844, spleen 0.834, GI
0.812, epidermis 0.811. Worst platforms: MIBI breast 0.63 (n=4), PDAC 0.68 (n=12).
For reference, Mesmer reported ~0.85–0.9 on this split. This is the number SAM has to
reach or explain.

Read the Xenium table as *agreement between two automated segmenters*, not accuracy: 10x's masks
are the reference, not ground truth. Cellpose finds ~10 % fewer cells than 10x (10x
splits more aggressively in dense tumor nests). ~5 s per ROI on the 4070 Ti. Human-
annotated accuracy comes from TissueNet (day 7). SAM 3.1 will be scored the same two
ways: vs 10x on these ROIs, and vs annotations on TissueNet.

## SAM 3 smoke test (done 2026-10-20 evening, `scripts/sam3_smoke.py`, roi01 centre crop)

HF access granted and token stored in the `spatial` env (NickA04). `sam3.pt` is cached
under `~/.cache/huggingface/hub`. Two things the README does not say but the notebooks
do: run under `torch.autocast("cuda", dtype=torch.bfloat16)` or the ViT MLP throws a
bf16/fp32 dtype error, and `.float()` the outputs before `.numpy()`.

Text prompt `"cell"` on the RGB composite (R boundary, G 18S, B DAPI), matched F1@0.5
vs 10x masks:

| crop (px) | cells in crop | threshold | SAM masks | F1 | P | R |
|---|---|---|---|---|---|---|
| 1024 | 632 | 0.3 | 116 | 0.18 | 0.59 | 0.11 |
| 512 | 169 | 0.3 | 127 | **0.55** | 0.64 | 0.48 |
| 512 | 169 | 0.1 | 184 | 0.48 | 0.47 | 0.49 |
| 256 | 40 | 0.3 | 48 | 0.38 | 0.36 | 0.40 |

`"blue blob"` ≈ `"cell"`; `"cells"` and `"nucleus"` return ~nothing. Model load 7 s,
inference 0.1–0.4 s per tile. Findings that shape days 2–3:
1. **Hard cap: `num_queries=200`** in the detector (model_builder.py:185). A tile can
   never return more than 200 instances, and in practice ~130 at threshold 0.3. Xenium
   density here is ~2,400 cells per 2048² ROI ≈ 150 per 512², so **tile at 512 px (or
   ~400 px) with overlap and stitch**, never 1024.
2. At 1024 the input is downscaled to 1008 and cells become ~20 px; recall collapses.
   512 → 1008 is a 2× upsample and works best of the three tried.
3. What SAM returns for "cell" is nucleus-sized blobs (DAPI), not 10x's whole-cell
   outlines, so IoU with 10x's membrane-bounded masks is penalised even when the
   detection is right. Options: score SAM against 10x *nuclei* too; prompt on a
   boundary-stain-only composite; or use SAM text detections as seeds and box-prompt
   (`add_geometric_prompt`) a whole-cell mask per seed. Decide on day 2 from data.
4. Lowering the threshold to 0.1 adds masks but not matches (precision falls, recall
   flat), so the misses are not low-confidence detections; they are absent queries.

## Tiled SAM 3 over all ROIs (done 2026-10-20, `bench/sam3_tiled.py`, `results/sam3_tiled.csv`)

512 px tiles, 64 px overlap, centroid-ownership stitching (each cell predicted by exactly
one tile), ~7 s per 2048² ROI (25 tiles). Mean over the 5 ROIs, matched F1@0.5:

| config (composite / prompt / tile / thr) | cell F1 | P | R | nucleus F1 | cells found vs 10x |
|---|---|---|---|---|---|
| **dapi_boundary / cell / 512 / 0.3** (best) | **0.511** | 0.53 | 0.50 | **0.692** | ≈ 96 % |
| dapi (grayscale) / cell / 512 / 0.3 | 0.511 | 0.55 | 0.48 | 0.615 | |
| rgb / cell / 512 / 0.3 (canonical, `cells_sam3.tif`) | 0.453 | 0.46 | 0.44 | 0.676 | 2,350 vs 2,442 |
| all / cell / 512 / 0.3 | 0.458 | | | 0.675 | |
| rgb / cell / 384 / 0.3 | 0.440 | 0.40 | 0.49 | 0.687 | |
| rgb / cell / 512 / 0.5 | 0.442 | 0.51 | 0.39 | 0.646 | |
| rgb / cell / 640 / 0.3 | 0.408 | | | 0.592 | |
| boundary (grayscale) / cell | 0.357 | | | 0.339 | |
| rgb / "whole cell" | 0.255 | | | | |
| rgb / "cell body", "cell membrane", "nucleus" | ≤ 0.05 | | | | |
| Cellpose cpsam (reference) | 0.782 | 0.84 | 0.74 | 0.706 | ≈ 90 % |

**Update, same evening (worked ahead into day 2):** the table above was produced with a
stitcher that painted higher-score masks over lower ones and left 21 % of instances
fragmented and 14 % as slivers. `stitch()` in `bench/sam3_tiled.py` now does duplicate
suppression (drop an instance if ≥ 50 % of it is covered by one accepted instance *and*
it is ≥ 50 % of that instance's area, so a nucleus nested in a two-nucleus blob is kept),
nearest-centroid assignment of contested pixels, largest-component + hole-fill per
instance, and tile ownership by midpoints between tile centres (`tile_cores`; the old
half-overlap rule double-owned a 256 px strip because the last tile column overlaps
its neighbour by 320 px). No tile reaches the 200-query cap (max 155 at 512 px).

Current numbers, mean of 5 ROIs vs 10x, matched F1@0.5:

| stage | config | cell F1 (P / R) | nucleus F1 | median area |
|---|---|---|---|---|
| SAM text seeds | dapi_boundary, "cell", **tile 384**, thr 0.3 | 0.565 (0.58 / 0.55) | **0.806** | 27 µm² |
| SAM text seeds | same, tile 512 | 0.559 | 0.751 | |
| SAM text seeds | same, tile 256 | 0.515 | 0.795 | |
| seeds → watershed into boundary stain (r 6 µm) | `bench/sam3_wholecell.py` | 0.608 (0.63 / 0.59) | | 46 µm² |
| **seeds → SAM box prompt, dilate 2 µm** | `bench/sam3_boxprompt.py` | **0.730 (0.75 / 0.71)** | | 45 µm² |
| seeds → box prompt, dilate 1.5 / 3 µm | | 0.717 / 0.699 | | 40 / 52 µm² |
| Cellpose cpsam (reference) | | 0.782 (0.84 / 0.74) | 0.706 | 44 µm² |

Per ROI, box-prompt d2 vs Cellpose: 0.769/0.824, 0.745/0.804, 0.738/0.790, 0.696/0.781,
0.700/0.713. Cost: ~12 s text pass (49 tiles) + ~23 s box pass (~2,300 boxes, batches of
64) per 2048² ROI. Ceiling check: watershed from *Cellpose's* nuclei reaches 0.805 and
from 10x's own nuclei 0.775, so with SAM nuclei at 0.81 the remaining gap is in the
whole-cell step, not detection.

**Canonical SAM pipeline now = text "cell" on R-boundary/B-DAPI at 384 px tiles →
box-prompt each seed with a 2 µm-dilated box (`model.predict_inst`, needs
`build_sam3_image_model(enable_inst_interactivity=True)`).** The box-prompt call is
also the repair primitive for the QC agent (re-prompt one cell with a corrected box or
with positive/negative points).

Original conclusions from the text-only sweep:
- **Text-only SAM 3 is a nucleus detector here.** Nucleus F1 0.69 is within 0.02 of
  Cellpose's nuclei (0.71) and it finds ~96 % of 10x's cell count. Whole-cell F1 is
  capped near 0.5 because the masks are nucleus-sized (median 26–31 µm² vs 10x's
  45 µm²), not because cells are missed.
- Prompt is `"cell"`; every other phrase is worse or empty. Tile 512, threshold 0.3.
- Composite: R = boundary stain, B = DAPI, G empty (`--composite dapi_boundary`) is the
  best input; adding 18S/αSMA to green hurts slightly; boundary stain alone fails.
- **Day-2 direction:** keep SAM text detections as the per-cell seed (nucleus + box),
  then get whole-cell masks by (a) box-prompting SAM with the seed box dilated to
  expected cell size, or (b) classical expansion of the SAM nucleus into the boundary
  stain (watershed on channel 1 with SAM seeds). Score both vs 10x cells; (b) is cheap
  and likely to close most of the 0.51 → 0.78 gap. Also score SAM nuclei vs 10x nuclei
  on TissueNet-style human labels (TissueNet has nuclear labels in `y[..., 1]`).

## Packaged pipeline (done 2026-10-21, `src/spatialprompt/`, `bench/sam3_pipeline.py`, `results/sam3_pipeline.csv`)

The two bench passes are now one module. `pip install -e .` in the `spatial` env
(already done; package `spatialprompt`, `pyproject.toml` at repo root).

```python
from spatialprompt import Segmenter, SegmentConfig, make_composite
seg = Segmenter()                       # loads facebook/sam3 once, interactive head on, bf16 autocast
rgb = make_composite(img4ch, "dapi_boundary")
res = seg.segment(rgb, SegmentConfig(pixel_size_um=0.2125))
res.cells, res.nuclei                   # uint32 labels, same ids (nucleus i is inside cell i)
res.stats                               # n_seeds, n_cells, detect_sec, expand_sec, mean_pred_iou, ...
seeds, st = seg.detect(rgb)                      # pass 1 only (text prompt, tiled, stitched)
cells, st = seg.expand(rgb, seeds, dilate_um=2)  # pass 2 only
masks, iou = seg.prompt_boxes(crop, boxes_xyxy)  # QC-agent repair primitive: N boxes -> N masks
```

- `SegmentConfig` defaults are the canonical config: prompt "cell", tile 384, overlap 64,
  threshold 0.3, dilate 2 µm, batch 64, `multimask=False`, `expand=True`.
- Images smaller than a tile are zero-padded (TissueNet 256 px tiles, QC crops).
- CLI: `spatialprompt-segment data/roi/roi01/image.tif` writes `cells_sp.tif`,
  `nuclei_sp.tif`, `segment_sp.json` next to it (reads `pixel_size_um` from `roi.json`).
  `python -m spatialprompt.segment` works too; the console script is not on PATH.
- `match_f1` moved to `spatialprompt.metrics`; `bench/cellpose_baseline.py` re-exports it.
- Regression check (`bench/sam3_pipeline.py`, defaults, all 5 ROIs): mean cell F1
  **0.725** (P 0.75 R 0.70), nucleus F1 **0.806**, 27 s/ROI, vs bench scripts 0.730 /
  0.806. Per ROI 0.769 / 0.741 / 0.731 / 0.691 / 0.694; the 0.005 difference is bf16
  run-to-run noise. Mean predicted IoU of the box masks is 0.81.
- Outputs per ROI: `cells_sp_<tag>.tif`, `nuclei_sp_<tag>.tif`, `overlay_sp_<tag>.png`.
- `multimask=True` (3 candidates per box, keep highest predicted IoU) is **worse**:
  roi01–03 0.758 / 0.736 / 0.729 vs 0.769 / 0.741 / 0.731 (stopped after 3 ROIs; predicted
  IoU rises to 0.84 but the picked masks drift away from 10x's boundaries). Keep `False`.
- **Threshold 0.2 on 384 px tiles is worse: 0.710** (P 0.71 R 0.71), nucleus 0.797; +7 %
  seeds are mostly false positives.
- **Seed-mask hint** (`seed_hint=True`: dilated seed mask fed as the dense prompt next to
  the box; the dense prompt must be 288×288 for SAM 3, `sam_prompt_encoder.mask_input_size`)
  is worse: roi01/02 0.748 / 0.739 vs 0.769 / 0.741, predicted IoU 0.94 but masks hug the
  nucleus; 79 s/ROI. Keep `False`.
- **Gap-closing verdict (day 2):** all four cheap ideas (multimask, threshold 0.2, seed
  hint, and on TissueNet smaller tiles / dilation) lose to the default. The 0.05 Xenium gap
  and 0.16 TissueNet gap are a seeds-pass recall limit, not a tuning problem. Stop tuning;
  the QC agent's targeted re-prompting is the designed route to recover missed cells.

| Xenium variant (`results/sam3_pipeline.csv`) | cell F1 | nuc F1 | s/ROI |
|---|---|---|---|
| **default** (384 / 0.3 / 2 µm) | **0.725** | 0.806 | 27 |
| threshold 0.2 | 0.710 | 0.797 | 25 |
| multimask (3 ROIs) | ~0.005 below default | | 30 |
| seed hint (2 ROIs) | ~0.012 below default | | 79 |
| Cellpose | 0.782 | 0.706 | |

- Do not run two SAM jobs on the 16 GB card at once: both load ~7 GB and thrash.

### TissueNet test, SAM pipeline vs human labels (2026-10-21, `bench/tissuenet_sam3.py`, `results/tissuenet_sam3_test.csv`)

Default config except tile 256 (= one TissueNet image, no stitching), composite R = membrane
channel, B = nuclear, pixel size from meta (0.5 µm for 1,080 of 1,324 images). 0.5 s/image.

| | SAM 3 cell F1 | Cellpose cell F1 | SAM nucleus F1 |
|---|---|---|---|
| **all 1,324** | **0.690** (P 0.78 R 0.63) | **0.854** (P 0.90 R 0.82) | 0.749 |
| tonsil | 0.795 | 0.918 | 0.805 |
| lung | 0.791 | 0.907 | 0.793 |
| breast | 0.701 | 0.862 | 0.759 |
| epidermis | 0.699 | 0.811 | 0.789 |
| pancreas | 0.693 | 0.844 | 0.752 |
| lymph node | 0.645 | 0.894 | 0.653 |
| GI | 0.604 | 0.812 | 0.715 |
| spleen | 0.502 | 0.834 | 0.070 |

- The gap is **recall**: SAM predicts ~77 % as many cells as the labels (n_pred/n_gt
  0.77 overall, 0.46–0.53 in lymph node / spleen with 260–300 cells per 256² tile). The
  200-query cap is never hit (max per tile < 190 everywhere); the text detector simply
  under-detects in dense fields at 256 px → 1008 px (4× upsampling, cells ~20 px).
- Worst at 0.365 µm/px (0.53, n=92) and 1.0 µm/px (0.57, n=16); 0.5 µm/px gives 0.71.
- Spleen nucleus F1 0.07 is a **label artefact**, not ours: spleen tiles have 186–244
  nuclear labels for 255–352 cell labels and only 61–88 % of nuclear-label pixels fall
  inside any cell label (every other tissue: ~100 %). Its nuclear channel is also
  8-bit-quantised with p99 ≈ 20 counts. Quote spleen cell F1 only, and never spleen nucleus F1.
- Lymph node is pure under-detection: tiles with 364–417 labelled cells get ~144 seeds
  (one 256 px tile → 1008 px); its nucleus F1 tracks cell F1, so the seeds pass is the limit.
- **Tile 128 / overlap 32 is worse: 0.633** (P 0.62 R 0.66), nucleus 0.684, 2.8 s/img.
  Recall rises slightly but precision drops 0.16: upsampling 128 px → 1008 px turns 20 px
  cells into 160 px blobs and the detector fires on fragments. The Xenium tile trick does
  not transfer; keep one 256 px tile per TissueNet image.
- **Threshold 0.2 is worse: 0.673** (P 0.72 R 0.65). Recall moves +0.014 for −0.055
  precision, so the missed cells are not low-confidence proposals; the text detector does
  not propose them at all. Both recall levers fail → the seeds pass is the ceiling here.
- **Dilate 1 µm is worse: 0.634** (P 0.72 R 0.58): tighter boxes shrink matched masks
  below IoU 0.5. Dilate 3 µm: 0.667, so the curve peaks at 2 µm (see table).
  After that, stop sweeping TissueNet: the honest number is ~0.69 vs Cellpose 0.85, and
  SAM's win is on Xenium nuclei (0.81 vs 0.71) plus promptability.

| TissueNet config (256 px, thr 0.3 unless noted) | cell F1 | P | R |
|---|---|---|---|
| **dilate 2 µm (default)** | **0.690** | 0.78 | 0.63 |
| threshold 0.2 | 0.673 | 0.72 | 0.65 |
| dilate 1 µm | 0.634 | 0.72 | 0.58 |
| tile 128 / overlap 32 | 0.633 | 0.62 | 0.66 |
| dilate 3 µm | 0.667 | 0.75 | 0.61 |

Dilation curve on TissueNet peaks at 2 µm (1 → 0.634, 2 → 0.690, 3 → 0.667), the same
optimum as Xenium. 3 µm loses 0.05 on breast and lung where cells are packed, and is flat
on tonsil and pancreas. The nucleus score is identical across dilations (seeds unchanged).

## Quantifier (done 2026-10-21, `src/spatialprompt/quantify.py`, `bench/quantify_check.py`, `results/quantify_check.csv`)

`quantify_transcripts(label, transcripts, pixel_size_um, genes=panel)` assigns each
transcript to the label covering its (x_px, y_px) pixel, keeping qv ≥ 20 and `is_gene`
exactly like 10x's cell_feature_matrix; returns a CSR cells × genes matrix, the gene
order, and a per-cell table (label, centroid px/µm, area, n_transcripts).
`marker_table(counts, genes, ["CD8A", ...])` gives dense marker columns and raises on a
gene that is not on the panel (PTPRC and ACTA2 are **not** on the 5K panel; CD45 and αSMA
exist only as stains). `intensity_table(label, img, names)` is the CODEX/stain analogue.

Validation on 10x's own masks (5 ROIs, ~2,400 cells each) vs 10x's cells.parquet:

| | all cells | interior cells (not touching ROI edge) |
|---|---|---|
| cells within 5 % of 10x's transcript_counts | 85 % | 88 % |
| cells within 10 % | 96 % | 98.7 % |
| per-cell total correlation r | 0.997 | |
| our total / 10x total | 0.966 | 0.977 |

- The remaining ~2 % deficit is rasterisation: our label rasters are 2.7 % smaller than
  10x's polygon areas (edge pixels lost when polygons were burned at 0.2125 µm), so
  membrane-adjacent transcripts fall outside. 80 % of interior diffs are in that
  direction. Acceptable; if it ever matters, dilate the 10x raster by one pixel in
  `scripts/extract_rois.py`. Cells touching the ROI edge are off because 10x counts their
  transcripts outside the crop (73 of the 103 cells off by > 10 % are border cells).
- Exact gene-vector matches are only 20 % of cells, for the same edge-pixel reason; this
  is not a bug.
- Transcript capture by mask set (fraction of qv ≥ 20 gene transcripts inside any cell,
  mean of 5 ROIs): **10x 89 %, Cellpose 81 %, SAM 71 %**. Median transcripts per cell are
  similar (83 / 83 / 80), so SAM loses capture through fewer and slightly tighter cells,
  not emptier ones. Relevant for the phenotype benchmark: gates on low-count markers
  (CD8A is ≥ 1 in only ~3 % of cells) will be a little less sensitive on SAM masks.
- Per-ROI outputs: `cells_10x_requant.parquet`, `cells_sp.parquet`, `cells_cellpose.parquet`
  (per-cell table + 10 marker columns).

## Query parser + phenotype gate (built 2026-10-21, `src/spatialprompt/query.py`, `gate.py`)

- **Meta Model API facts (dev.meta.ai/docs, fetched 2026-10-21):** base URL
  `https://api.meta.ai/v1`, key in env `MODEL_API_KEY` sent as a Bearer token, current
  model id `muse-spark-1.3` (also 1.2, 1.1; `-contributor` variants). Tool calling incl.
  parallel calls is supported; structured output is listed but the JSON-mode mechanics
  are unverified. `openai` 3.36 is installed in the `spatial` env. **Put the key in
  `.env` as `MODEL_API_KEY=...`** (query.py also accepts `META_API_KEY`).
- `parse_query(text, panel, backend="auto")` → query dict. Backend "auto" uses Muse Spark
  when the key is set and falls back to the **rules backend** (vocabulary of 10 cell
  types, 3 regions, 4 spatial relations, µm distances, `GENE+`/`GENE-` syntax) otherwise.
  Muse output is validated against the panel (unknown genes dropped with a warning).
- **Muse backend works (2026-10-26, personal key, `bench/query_muse_vs_rules.py`,
  `results/query_muse_vs_rules.csv`).** 12 demo queries: same markers as rules on 3/12, same
  region + relation on 11/12 (the 12th is only "vessels" vs "vessel"). Where they differ Muse
  is better biology: Tregs → CD3 & CD4 & FOXP3, "HER2+" adds an ERBB2 group, "proliferating"
  adds MKI67/TOP2A/PCNA/CCNB1, "exhausted CD8" adds (PDCD1|HAVCR2|LAG3|TIGIT|TOX), macrophage /
  endothelial groups get CSF1R/MRC1 and CLDN5/KDR/PLVAP. It avoids genes listed as absent.
  ~$0.005 per query on `muse-spark-1.3` (mostly ~1k reasoning tokens at effort "low").
- **All Muse calls go through `spatialprompt.muse.chat`**: disk cache (`data/cache/muse/`),
  per-key spend ledger (`data/logs/muse_usage.jsonl`, keyed by a sha256 fingerprint) and a
  hard cap `MUSE_BUDGET_USD` checked against the worst case *before* sending. `python -m
  spatialprompt.muse` prints spend; `--ping` makes a minimal call. `.env` holds
  `MODEL_API_KEY`, `MUSE_BUDGET_USD`, `MUSE_MODEL`. Currently a **personal key**, budget $10;
  the Meta hackathon key ($150 credits) gets its own ledger automatically.
- API facts verified by calls: `response_format` **`json_schema`** works (the docs list only
  that type; nullable objects via `"type": ["object", "null"]` are accepted). Reasoning cannot
  be disabled (`"none"` → 400), effort `minimal|low|medium|high|xhigh|max`. **`completion_tokens`
  includes `reasoning_tokens`** (ping: completion 23 = 13 reasoning + visible). `max_tokens`
  caps both, so a long reasoning trace can truncate the JSON: `parse_muse` retries once at
  4000. Prices per 1M: standard $1.25 in / $0.15 cached / $4.25 out; `-contributor` ids
  $0.10 / $0.002 / $0.20, but Meta may train on those requests. Rate limits are per team.
- Bugs fixed with the Muse work: the panel hint used to be `sorted(panel)[:400]` (genes A–B
  only); it is now the on/absent split of ~130 canonical markers (~230 tokens). An empty Muse
  answer used to be accepted for any query containing "cell"; now only when the rules backend
  also finds no markers (a generic "all cells").
- Query schema: `positive` = AND of OR-groups (sum ≥ `min_count`), `negative` groups
  (sum < 1), `region` (marker rule for the reference tissue), `spatial`
  (`boundary | inside | outside | within`, `distance_um`), `sam_prompt` (always "cell").
- `gate(table, counts, genes, query, label, px)` → boolean keep + info. Spatial filter:
  region cells' masks → binary → closing 5 µm + fill holes → signed distance of each cell
  centroid to the region edge.
- End-to-end on roi01 (rules backend, all three mask sets):

| query | 10x | SAM | Cellpose |
|---|---|---|---|
| CD8+ T cells (marker only) | 96 | 96 | 95 |
| … at the tumor boundary (±20 µm) | 48 | 39 | 44 |
| … inside the tumor | 3 | 0 | 0 |
| tumor cells (region, ≥3 of 6 markers) | 365 | 298 | 358 |
| macrophages | 114 | 98 | 102 |

  Overlay `results/gate/roi01_sam_cd8t_tumor_boundary.png` (cyan = tumor region edge,
  magenta = kept, yellow = CD8 T not kept): kept cells line the nest edges; this ROI is
  immune-excluded (almost no CD8 T inside nests), consistent with the overlay.
- SAM finds 18 % fewer tumor cells than 10x (298 vs 365) — the recall gap lands on the
  dense epithelial nests, exactly where the QC agent should look for missed cells.

## QC agent (built 2026-10-21, `src/spatialprompt/qc/`, `bench/qc_check.py`, `results/qc_check.csv`)

Design: `detectors.py` (plain functions over label images), `repair.py` (add via box
prompt, split via two box prompts, merge, drop), `agent.py` (`QCAgent.run` = detect →
plan → repair → re-detect; `RulesPlanner` today, `MusePlanner` stub with the same
`plan(flags, context) -> actions` interface for Muse Spark later). Evidence is the DAPI
foreground (smoothed, fraction of Otsu) compared with the cells and their seeds; SAM
cells have exactly one seed each by construction, so "extra nucleus" = DAPI foreground
inside a cell away from its seed.

**What the SAM recall gap actually is (roi01, 614 of 2,442 10x cells unmatched at IoU 0.5):**
- 343 have **no SAM seed** inside: 136 mostly uncovered by any SAM cell, 82 partly covered,
  125 swallowed by a neighbouring SAM cell (a doublet from SAM's side).
- 271 **have a seed but IoU < 0.5** (outline disagreement); 183 of them sit at IoU 0.4–0.5.
- 147 SAM cells hold ≥ 2 10x nucleus centroids (true doublets); 194 hold none.
- Uncovered misses are **dim**: median smoothed DAPI 0.13 vs 0.27 for all 10x cells. At
  0.6 × Otsu only 46 % of them show ≥ 10 µm² of foreground; at 0.3 × Otsu 88 % do.
- True doublets look like ordinary cells (area 52 vs 46 µm², ≤ 1 µm² of foreground outside
  the seed zone) and DAPI peak counting does not separate them (45 % vs 34 % with ≥ 2
  peaks). SAM's seed already spans both 10x nuclei. **Doublets are not detectable from
  DAPI alone**; the doublet detector keeps precision ~0.6 at recall ~0.05.

**Lessons from the first versions (keep):** distance-transform watershed nuclei from DAPI
score only 0.44 F1 vs 10x nuclei and made every detector fire wrongly (488 "empty", 191
"split" flags on roi01; applying them took F1 0.769 → 0.620). Merges and drops are now
**off by default** in `RulesPlanner`; only adds and splits run.

**Add-cell sweep (roi01 + roi03, add-only):** F1 before 0.750. Best 0.754 at 0.45 × Otsu
with a 120 µm² cap on the added mask (≈100 adds, recall +0.02, precision −0.013).
0.3 × Otsu adds 133 cells and loses F1 (0.745). Gating adds on SAM's predicted IoU ≥ 0.8
leaves ~28 adds and gains nothing; ≥ 0.9 leaves none. Defaults set accordingly.

**Five-ROI result (`bench/qc_check.py`, RulesPlanner, one round, ~28 s/ROI):**

| | mean of 5 ROIs |
|---|---|
| missed detector | P 0.55, R 0.27 of truly-missed 10x cells (~190 flags/ROI, ~100 adds applied) |
| doublet detector | P 0.53, R 0.03 (5–17 flags/ROI) |
| empty detector | P 0.71, R 0.34 (flags only; drops are off) |
| cell F1 vs 10x | **0.725 → 0.728** (add-only 0.729, split-only 0.724) |
| recall / precision | 0.704 → 0.723 / 0.747 → 0.732 |

Per ROI the gain is +0.001 to +0.005, never negative. **Verdict:** the repair loop is
safe and recovers a few percent of the missed cells, but it does not close the gap. The
cells SAM's text pass skips are dim nuclei that the box prompt also outlines poorly (added
masks often miss IoU 0.5 against 10x), and the 125–200 swallowed nuclei per ROI are
invisible to DAPI-only evidence. A Muse Spark planner can only choose among the same
repairs, so do not expect it to change these numbers; its value is explaining and
prioritising flags for a human, not raw F1. Overlay: `data/roi/roi01/overlay_sp_qc.png`.
(Superseded in part by the MusePlanner result below: with crops, Muse *does* change the
numbers, by rejecting bad flags rather than by finding new repairs.)

### MusePlanner (built 2026-10-26, `src/spatialprompt/qc/muse_planner.py`, `bench/qc_check.py --planner muse`, `results/qc_check_muse.csv`, `results/qc_flags/`)

Muse Spark 1.3 as a **triage judge over the detectors' proposals**, looking at each flagged cell:
- Per flag a two-panel crop (40 µm field, 2× upscaled, 376×752 px): left DAPI grayscale, right
  the R-boundary/B-DAPI composite; overlays = current cell outlines (yellow), the flagged DAPI
  structure (green box), the flagged cell itself (cyan; doublet/empty). `empty` and `doublet`
  rows now carry the cell bbox from `detect_all` for this.
- 16 flags per call (one text line of numbers + one image each), `json_schema` answer
  `{"decisions": [{flag_id, action ∈ add|split|drop|skip, confidence, reason ≤ 12 words}]}`,
  reasoning effort "low", `max_tokens` 3000 (retry 6000). Accepted decisions (confidence ≥ 0.6)
  become the same action dicts RulesPlanner emits; `drop` is applied only when Muse confirms it.
  No tool calling: the Meta API supports `tool_choice: "auto"` only, so a call cannot be forced.
- Everything run-specific stays out of the messages, so re-runs hit the cache. Decisions,
  reasons and cost land in `qc_log_muse.json` (`log["planner"]`) and per flag in
  `results/qc_flags/<roi>_muse.csv` joined to the 10x truth.
- Budget estimator in `muse.py` counts an image part as a flat 400 tokens (its base64 would
  otherwise be "budgeted" as ~24k tokens and trip the cap). `bench/qc_muse_smoke.py` runs one
  call without loading SAM.
- **The prompt is the whole game.** First prompt ("skip dim debris, be conservative") rejected
  everything: 2/36 missed accepted, 0/12 doublets; its reasons were "dim haze, no bright
  nucleus" on 15 of the 18 true misses it skipped, which is exactly what CLAUDE.md records
  about the misses (dim nuclei), and "single lobulated nucleus" on true doublets (10x splits on
  the 18S stain Muse cannot see). Recalibrated prompt (dim nucleus-sized blobs with their own
  territory are cells; a 10x doublet looks like one bilobed DAPI region; "give the action you
  believe") on the same 36 flags per kind: missed accept-precision 0.76, doublet 2/12 both
  true, empty 0.65.

roi01, all 287 flags (188 missed, 12 doublet, 87 empty), 18 calls, **$0.30**, 177 s:

| kind | Muse accepted | accept-precision (rules = detector) | true flags kept | rejection-precision |
|---|---|---|---|---|
| missed | 93 / 188 | **0.68** (0.48) | 0.70 | 0.72 |
| doublet | 3 / 12 | 0.67 (0.58) | 0.29 | 0.44 |
| empty | 49 / 87 | **0.80** (0.56; rules never drops) | 0.80 | 0.74 |

Cell F1 vs 10x: before 0.769 → **Muse 0.778** (P 0.80 R 0.76) vs RulesPlanner 0.770.
Applied: 54 adds, 49 drops, 3 splits. The gain comes from *rejecting* half the missed flags and
from drops, which were unusable without a judge.

**Five ROIs, all flags, batch 16 (`results/qc_check_muse.csv`; rules numbers from
`results/qc_check.csv`), 99 calls, $1.62, ~3.2 min per ROI (vs 30 s for rules):**

| ROI | F1 before | rules | **Muse** | missed acc-P (det.) / kept | empty acc-P (det.) / kept | doublets accepted | calls / $ |
|---|---|---|---|---|---|---|---|
| roi01 | 0.769 | 0.770 | **0.778** | 0.68 (0.48) / 0.70 | 0.80 (0.56) / 0.80 | 3 / 12 | 18 / 0.30 |
| roi02 | 0.741 | 0.746 | **0.763** | 0.73 (0.58) / 0.75 | 0.95 (0.77) / 0.90 | 1 / 5 | 22 / 0.36 |
| roi03 | 0.731 | 0.735 | **0.743** | 0.65 (0.57) / 0.54 | 0.92 (0.72) / 0.84 | 3 / 12 | 17 / 0.28 |
| roi04 | 0.691 | 0.693 | **0.712** | 0.67 (0.55) / 0.50 | 0.95 (0.81) / 0.97 | 2 / 17 | 24 / 0.39 |
| roi05 | 0.694 | 0.695 | **0.705** | 0.78 (0.59) / 0.72 | 0.94 (0.66) / 0.75 | 1 / 6 | 18 / 0.28 |
| **mean** | 0.725 | 0.728 | **0.740** (P 0.77 R 0.72) | 0.70 (0.55) / 0.64 | 0.91 (0.71) / 0.85 | 2 / 10 | 20 / 0.32 |

- Muse beats RulesPlanner on **every** ROI (+0.008 to +0.017 over rules, +0.011 to +0.022
  over no QC); precision rises 0.747 → 0.766 and recall 0.704 → 0.716. Rules only ever got
  +0.003. Per ROI ~95 adds accepted of ~190 flags (≈ 55 applied), ~77 drops, 1–3 splits.
- Where the gain comes from: (a) rejecting the ~45 % of missed flags that are not cells, so
  the adds that go in are 70 % right instead of 55 %; (b) drops, which no planner could risk
  before (detector precision 0.71) and which Muse confirms at 0.91 while keeping 85 % of the
  true empties. Doublets remain out of reach (2 of ~10 flags accepted, and the detector's
  recall is 0.03 anyway), as predicted: 10x splits on the 18S stain Muse is not shown.
- Honest framing for the write-up: the judge improves the *decision quality* a lot (missed
  0.55 → 0.70, empty 0.71 → 0.91) and F1 a little (+0.015), because F1 is bounded by the
  repair primitive (box-prompted adds of dim nuclei often land below IoU 0.5 vs 10x) and by
  the detectors' recall (0.27 for misses). The agentic story is real: detect → look → decide
  with a reason → repair, and every decision is logged with its reason.
- Cost/time: $0.32 and ~3 min per 2048² ROI at ~20 calls of 16 crops; reasoning is ~70 % of
  output tokens. One roi04 batch hit `max_tokens` 3000 on reasoning and was retried at 6000.
  Cached reruns are free. Possible phase 2 (not started): verify pass with id-based undo; an
  18S panel in the crop for doublets; `reasoning_effort="minimal"` to halve cost.
- Spend after this: **$1.90** (ledger) / $1.77 (Meta's balance, $148.23 of the $150 hackathon
  credit, which landed on this same key on 2026-10-26). **Spend policy (Nick, 2026-10-26): very
  conservative until the product works end to end.** `.env` cap `MUSE_BUDGET_USD=4.00`
  (raise deliberately, per job, after stating the estimate); `MUSE_OFFLINE=1` for rehearsals
  (cache-only, $0); no new Muse evaluation runs (phase-2 ideas, more ROIs, prompt sweeps)
  without asking first. A new demo query costs ~$0.005; a Muse QC pass on one ROI ~$0.32.

## End-to-end CLI + phenotype benchmark (done 2026-10-26, day 8)

`spatialprompt-run --roi data/roi/roi01 --query "CD8+ T cells at the tumor boundary" [--masks sam+muse]`
(`src/spatialprompt/cli.py`; `python -m spatialprompt.cli` works too). Steps, each timed: masks
(reused from the ROI dir: `--masks sam | sam+qc | sam+muse | 10x | cellpose`; `--fresh` recomputes
SAM / QC, Segmenter loaded lazily) → quantify → parse (`--backend auto|muse|rules`) → gate →
write `results/runs/<roi>_<query-slug>_<masks>/`: `query.json`, `cells.parquet` (every cell, the
query's genes as columns, `marker_pass`, `keep`, `dist_to_region_um`, slide µm), `labels_query.tif`
(kept cells only), `masks_query.geojson` (polygons in slide µm, Xenium frame, properties = the
row), `overlay.png` (cyan region edge, magenta kept, yellow marker-positive not kept),
`summary.json`. roi01 demo query on cached masks: **9 s, 40 cells of 2,318, $0** (parse cached).
`results/runs/` is gitignored (24 MB per run); `results/figures/demo_overlay_roi01_cd8t_boundary.png`
is the 1024 px copy. `MUSE_OFFLINE=1` makes the whole CLI zero-spend.

**Web UI (done 2026-10-26, `src/spatialprompt/app.py`, Gradio 6.30 in the `spatial` env,
optional dep `pip install -e ".[app]"`):** `python -m spatialprompt.app` → http://127.0.0.1:7860.
Three tabs, no logic of its own: **Query** (ROI, mask set, parser backend, offline switch =
`MUSE_OFFLINE`; Parse shows Muse's JSON + cost; Run calls `cli.run` → overlay, summary, kept-cell
table, downloads), **QC agent** (every Muse decision for the ROI from `qc_log_muse.json` with
confidence/reason/applied; click a row → crop with masks before (cyan) / after (yellow) QC),
**Benchmarks** (the result CSVs + `results/figures/`). Verified in Chrome: demo query runs in
~10 s from cached masks, $0 in offline mode. Launches on 127.0.0.1 only.

**Phenotype benchmark** (`bench/phenotype_eval.py`, `results/phenotype_eval.csv`, 5 ROIs ×
10 queries × 4 mask sets, no API calls: Muse parses from cache, uncached → rules).
Reference = gate on 10x masks with 10x's own matrix; candidate kept cell matched to the 10x
cell under its centroid; P/R/F1 of the kept sets. `10x-requant` = 10x masks + our quantifier.

| query (reference cells / ROI) | 10x-requant | Cellpose | SAM | **SAM + Muse QC** |
|---|---|---|---|---|
| B cells (266) | 0.99 | 0.88 | 0.87 | **0.88** |
| tumor cells (255) | 0.98 | 0.83 | 0.76 | **0.76** |
| macrophages (156) | 0.98 | 0.74 | 0.73 | **0.74** |
| HER2+ tumor cells (99) | 0.98 | 0.78 | 0.69 | **0.68** |
| CD8+ T cells (71) | 0.98 | 0.82 | 0.83 | **0.83** |
| endothelial cells (64) | 0.99 | 0.78 | 0.76 | **0.77** |
| proliferating tumor cells (35) | 0.98 | 0.70 | 0.53 | **0.54** |
| CD8+ T cells at the tumor boundary (25) | 0.99 | 0.83 | 0.80 | **0.81** |

- Quantification is not the bottleneck (0.98–0.99 on 10x's own masks); the gap is segmentation
  agreement, and it tracks the mask F1 (Cellpose 0.78 > SAM+Muse 0.74 > SAM 0.725). SAM+Muse ≥
  SAM on 7 of 8 queries (HER2+ −0.006). Immune / stromal queries are close to Cellpose
  (−0.00 to −0.04); the tumor-nest queries (HER2+, proliferating) lose 0.10–0.16 because that
  is exactly where SAM under-segments dense epithelium.
- Rows with ≤ 1 reference cell per ROI are excluded: "CD8+ T inside the tumor" (immune-excluded
  ROIs) and "regulatory T cells near the tumor".
- **Schema change (2026-10-26, ~$0.13 of re-parsing):** positive groups are now
  `{"genes": [...], "min_count": n}` (plain lists still accepted, using the query's
  `min_count`); `gate.marker_mask` / `group_genes` / `group_min` handle both. First attempt told
  Muse "≥ 3 genes → min_count 3", which it applied to immune/stromal groups too (macrophages
  187 → 7 cells); the instruction is now "3 only for the epithelial signature, 1 for everything
  else". Rules backend emits the same shape and maps `HER2+ / Ki67+ / PD-1+` qualifiers through
  `GENE_ALIASES`. "HER2+ tumor cells" went from 5 to 99 reference cells per ROI.
- Figure for the deck not yet made (do it with `bench/make_figures.py` on day 9).

## Figures and deck (2026-10-21)

- `results/figures/`: `xenium_f1.png`, `tissuenet_f1.png`, `tissuenet_variants.png`,
  `capture_gate.png`, plus overlay crops. Regenerate from the CSVs with
  `bench/make_figures.py` (the roi01 gate counts are hard-coded there until a gate bench exists).
- Progress deck (12 slides, exportable to PowerPoint from its Share menu):
  https://claude.ai/artifact/FDmfqnwABXe1jzPWbWGqqX — private until shared.

## Ten-day plan (2026-10-20 → 2026-10-30)

| Day | Goal | Done when |
|---|---|---|
| 1 | ~~Xenium download, ROI picking, ROI extraction, Cellpose baseline on ROIs and TissueNet, SAM 3 install + HF access + smoke test~~ **done**. Meta developer account created, API key pending. Muse Code not installed. | ~~`scripts/sam3_smoke.py` runs a text prompt on a crop and saves masks~~ ✔ |
| 2 | ~~Tiled SAM over whole ROIs, compositor/prompt/tile sweep, stitcher fix, box-prompted whole-cell masks~~ **done on day 1 evening** (SAM 0.730 vs Cellpose 0.782 on Xenium). ~~Wrap the two passes into `src/spatialprompt/segment.py`~~ **done 2026-10-21** (reproduces 0.725 / 0.806). ~~SAM pipeline on TissueNet test vs human labels~~ **done 2026-10-21**: 0.690 vs Cellpose 0.854, four variants swept, none better (table in the TissueNet subsection). Gap-closing on Xenium: multimask worse; 0.2 threshold and seed-mask hint results in the Packaged pipeline section. | `results/tissuenet_sam3_test.csv` ✔ |
| 3 | ~~SAM segmentation on ROIs; label image output; quick visual check~~ **done** (`cells_sp_*.tif`, `overlay_sp_*.png` per ROI). Freed day → start quantifier early. | overlay PNGs per ROI ✔ |
| 4 | ~~Quantifier: transcripts → per-mask counts~~ **done 2026-10-21 (day 2)**: 96 % of cells within 10 % of 10x's counts on 10x's masks, r 0.997 (Quantifier section). | per-cell table matches 10x within tolerance ✔ `results/quantify_check.csv` |
| 5 | ~~Query parser + phenotype gate~~ **built 2026-10-21 (day 2)** with a rules backend; "CD8+ T cells at the tumor boundary" → 39–48 cells lining the nest edges on roi01. **Open:** exercise the Muse Spark backend once `MODEL_API_KEY` is in `.env`, and compare its JSON to the rules output on ~10 queries. | ~~sensible subset; boundary filter works~~ ✔ (rules); Muse comparison pending key |
| 6 | ~~QC agent: detectors as tools, repair loop with box re-prompts~~ **built 2026-10-21 (day 2)** with `RulesPlanner`: F1 0.725 → 0.728, safe on every ROI, missed-detector P 0.55 R 0.27, doublets undetectable from DAPI (QC agent section). ~~**Open:** `MusePlanner` once a key exists~~ **done 2026-10-26**: Muse judges each flag from a crop; roi01 F1 0.769 → 0.778 vs rules 0.770, missed accept-precision 0.48 → 0.68, drops usable (MusePlanner section). | before/after per ROI in `results/qc_check.csv` ✔; `results/qc_check_muse.csv` ✔ |
| 7 | ~~TissueNet benchmark harness: Cellpose vs SAM 3.1, F1@0.5~~ **done 2026-10-21 (day 2)**: `bench/tissuenet_sam3.py`, SAM 0.690 vs Cellpose 0.854 on 1,324 test images, variants all worse (TissueNet subsection). | table ✔ `results/tissuenet_sam3_test.csv`; bar chart ✔ `results/figures/tissuenet_f1.png` |
| 8 | ~~Phenotype benchmark on Xenium; end-to-end CLI `query → outputs`~~ **done 2026-10-26**: `spatialprompt-run` (9 s on cached masks, 6 outputs incl. GeoJSON); `bench/phenotype_eval.py` 5 ROIs × 10 queries (SAM+Muse F1 0.74–0.88 vs 10x on real populations; Cellpose +0.02–0.07). Open: per-group `min_count` in the query schema (~$0.05 to re-parse). | ~~one command runs the whole loop on an ROI~~ ✔ |
| 9 | Demo polish: ~~web UI~~ **done 2026-10-26** (Gradio app, 3 tabs); hybrid seeds tried, not adopted (0.737 vs Cellpose 0.782); query schema per-group thresholds. Still to do: README, deck figures (phenotype + QC tables), cold-start timing, video script. | demo runs cold in < 10 min |
| 10 | Write-up, video/screens, buffer | submitted |

Cut list if behind: CODEX (drop), per-platform TissueNet breakdown (drop), spatial
"boundary" filter (replace with "within N µm of a tumor-marker-positive cell" or drop),
QC repair (keep flagging, drop re-prompting).

## Repo layout (✔ = exists as of 2026-10-21, rest planned)

```
meta_spatialomics/
  CLAUDE.md             ✔
  README.md               # user-facing: what it does, how to run the demo
  pyproject.toml        ✔ # package "spatialprompt", py3.12, installed editable in env `spatial`
  env/                    # environment.yml / uv lock; SAM 3.1 pinned
  data/                 ✔ # gitignored; raw/ (downloads), roi/ (crops), tissuenet/, logs/
  vendor/sam3/          ✔ # gitignored clone of facebookresearch/sam3, pip install -e
  scripts/              ✔ # fetch_xenium.py, pick_rois.py, extract_rois.py, sam3_smoke.py
  src/spatialprompt/
    io/                   # xenium.py, codex.py, tissuenet.py (readers → common ROI object)
    composite.py        ✔ # marker channels → RGB (make_composite for Xenium 4-ch, rgb_from_channels general)
    segment.py          ✔ # Segmenter: detect (text, tiled) / expand (box) / prompt_boxes / segment; CLI
    metrics.py          ✔ # match_f1
    quantify.py         ✔ # quantify_transcripts / marker_table / intensity_table
    query.py            ✔ # parse_query: Muse Spark (Meta Model API) or rules backend → query dict
    gate.py             ✔ # gate: marker logic + spatial relation to a marker-defined region
    muse.py             ✔ # chat(): Meta Model API client, disk cache, per-key spend ledger + hard cap
    qc/                 ✔ # detectors.py, repair.py, agent.py (QCAgent, RulesPlanner), muse_planner.py (MusePlanner)
    cli.py              ✔ # spatialprompt-run: query -> masks, counts, gate, GeoJSON, overlay
    app.py              ✔ # Gradio demo UI (python -m spatialprompt.app)
    quantify.py           # transcripts or intensities per mask
    query.py              # Muse Spark: plain English → structured query
    gate.py               # marker logic + spatial filters
    qc/                   # detectors.py (doublet/split), agent.py (Muse Spark loop + tools)
    cli.py                # `spatialprompt run --roi ... --query "..."`
  bench/                ✔ # *_baseline / sam3_* / tissuenet_* / quantify_check / qc_check (+ qc_truth, qc_muse_smoke) / query_muse_vs_rules
  notebooks/              # demo walkthrough
  results/                # figures, tables, timings (small files committed)
  muse_sessions/          # copies of Muse Code JSONL logs for the write-up
```

## Conventions

- Python 3.12 for the SAM env. Type hints, `pathlib`, no notebooks in `src/`.
- Images: numpy `(C, Y, X)` float32 in [0, 1] internally; label images `uint32`, 0 =
  background. Masks go to disk as label TIFF + GeoJSON polygons (Xenium coordinate frame,
  microns, with the pixel size from the OME metadata).
- Every ROI has a `roi.json` (dataset, slide, x0, y0, w, h, pixel_size_um, channels).
- Model calls (SAM, Muse Spark) are cached to disk keyed by input hash so re-runs are free.
- Benchmarks write a CSV plus a figure; never hand-copy numbers into the write-up.
- Secrets: `META_API_KEY` (name TBC from docs) and `HF_TOKEN` in `.env`, gitignored.
- Windows: paths through `pathlib`; long-running GPU jobs from Git Bash or WSL.

## Open questions (resolve day 1–2, then record answers here)

1. ~~Does Muse Spark accept image input?~~ **Yes** (verified by calls 2026-10-26): Chat
   Completions content parts `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`,
   ≤ 50 images per request, ≤ 50 MB each, png/jpeg/gif/webp; cost scales with resolution
   (docs: ~3 tokens at 1×1, ~1,400 at 1280 px; our 376×752 two-panel crops cost ~250–300).
   `json_schema` structured output and 16 images in one request work together. The QC agent
   sees crops (see MusePlanner).
2. ~~Exact checkpoint loading call~~ `build_sam3_image_model()` loads sam3.pt; box
   prompts go through `Sam3Processor.add_geometric_prompt(box, label, state)` after a
   text prompt and are combined with it in the same forward pass.
3. Which Xenium breast ROIs have a clear tumor/stroma boundary with CD8A+ cells? Pick by
   browsing whole-slide `cells.parquet` + `cell_feature_matrix.h5` (download those two
   first, 0.1 GB) before pulling image members. The Zenodo crop region is immune-poor.
4. ~~Does SAM 3 segment single cells in a fluorescence composite at all?~~ Yes, at 512 px
   tiles (F1 0.55 vs 10x on the first try, untuned). See "SAM 3 smoke test".
5. ~~Where does this get submitted, and in what format?~~ Meta Global AI Developer
   Hackathon, CV platform, public repo + 3-min video, due 2026-11-04 9 PM ET (Hackathon section).
6. **Open:** how to treat the work done Oct 20–25 under the "new work only" rule, and which
   track (Indie Developer vs Startups) to enter.

## Hybrid seeds: tried 2026-10-26, NOT adopted (`bench/sam3_hybrid.py`, `results/sam3_hybrid.csv`)

Same `Segmenter.expand` (box prompt, dilate 2 µm) from four seed sets, 5 ROIs, whole-cell
F1 vs 10x (Cellpose end-to-end reference 0.782):

| seeds | n seeds | cell F1 (P / R) | nucleus F1 | per ROI |
|---|---|---|---|---|
| SAM text (control) | 2,275 | 0.725 (0.75 / 0.70) | **0.806** | 0.769 / 0.741 / 0.731 / 0.691 / 0.694 |
| Cellpose nuclei | 2,352 | 0.759 (0.77 / 0.75) | 0.706 | 0.786 / 0.779 / 0.746 / 0.750 / 0.732 |
| SAM ∪ Cellpose nuclei not within 1 µm of a SAM seed | 2,525 (+250) | 0.736 (0.72 / 0.75) | 0.806 | 0.775 / 0.751 / 0.736 / 0.713 / 0.706 |
| same, 2 µm margin | 2,491 (+216) | 0.737 (0.73 / 0.75) | 0.808 | 0.775 / 0.753 / 0.740 / 0.712 / 0.707 |

- Adoption rule was mean > 0.782 and ≥ Cellpose on 4/5 ROIs: fails on every ROI. The union
  buys +0.05 recall and loses 0.03 precision (the added Cellpose nuclei are partly the dim
  cells SAM skipped and partly duplicates/debris); net +0.012, about what the Muse QC gives.
- The useful finding: **Cellpose's own nuclei through SAM's box prompt reach 0.759, below
  Cellpose's 0.782 from the same nuclei.** So SAM's whole-cell outlines trail Cellpose's by
  ~0.02 even with identical seeds; the earlier "watershed from Cellpose nuclei = 0.805"
  ceiling was watershed into the boundary stain, not SAM. The limit is now the outline step,
  not detection, and no seed trick closes it.
- Verdict for the write-up: on whole-cell F1 vs 10x we do not pass Cellpose (0.740 with Muse
  QC vs 0.782); we pass it on Xenium **nucleus** F1 (0.81 vs 0.71) and we are the only one of
  the three that is promptable and returns phenotyped cells. `cells_sp_hybrid.tif` /
  `nuclei_sp_hybrid.tif` (2 µm union) stay on disk per ROI; not wired into the CLI. Untested:
  union + Muse QC (~$0.35/ROI), expected ~0.75.

## Parked ideas (do not start until the current design has produced its results)

## Sources

- SAM 3 paper: https://arxiv.org/abs/2511.16719 ; model page https://dev.meta.ai/models/sam-3-1/
- SAM 3.1 blog: https://ai.meta.com/blog/segment-anything-model-3/
- Muse Spark 1.1 announcement: https://ai.meta.com/blog/introducing-muse-spark-meta-model-api/
- TissueNet / Mesmer: https://doi.org/10.1038/s41587-021-01094-0
- Xenium breast Prime 5K: https://www.10xgenomics.com/datasets/xenium-prime-ffpe-human-breast-cancer
