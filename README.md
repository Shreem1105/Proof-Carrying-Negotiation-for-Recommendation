# PCN-Rec: Proof-Carrying Negotiation for Recommendation (Step 1)

Foundational codebase for the WWW'26 workshop short paper project.
This Step 1 implementation includes dataset preparation, candidate generation (LightFM), and a baseline MMR reranker.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration**
   Defaults are in `config/config.yaml`. You can modify them or override via CLI args (not fully supported in step 1 scripts yet, mostly config file driven).

## Running Step 1 Pipeline

Run the following scripts in order. All outputs go to `outputs/{run_id}/`.

1. **Prepare Data** (Downloads MovieLens, splits, computes popularity)
   ```bash
   python scripts/step1_prepare_data.py --config config/config.yaml --run_id test1
   ```

2. **Train Candidate Model** (LightFM)
   ```bash
   python scripts/step1_train_candidates.py --config config/config.yaml --run_id test1
   ```

3. **Generate Candidates** (Top-100 per user)
   ```bash
   python scripts/step1_generate_candidates.py --config config/config.yaml --run_id test1
   ```

4. **Run MMR Baseline** (Reranking)
   ```bash
   python scripts/step1_run_mmr_baseline.py --config config/config.yaml --run_id test1
   ```

5. **Smoke Test** (End-to-End on small subset)
   ```bash
   python scripts/smoke_test_step1.py
   ```

## Colab Usage
The scripts are designed to be runnable on Google Colab.
For large runs, ensure you have the `outputs` directory mounted or saved periodically.
For the smoke test, the defaults are small enough for any CPU.
