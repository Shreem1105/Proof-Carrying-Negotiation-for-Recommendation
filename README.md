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

## Step 2 Usage (PCN-Rec)

1. **Set API Key**
   ```bash
   export GEMINI_API_KEY=your_key_here
   # Or on Windows PowerShell:
   # $env:GEMINI_API_KEY="your_key_here"
   ```

2. **Run Single LLM Baseline**
   ```bash
   python scripts/step2_run_single_llm_baseline.py --config config/config.yaml --run_id exp1 --max_users 200
   ```

3. **Run PCN-Rec (Negotiation + Verifier)**
   ```bash
   python scripts/step2_run_pcnrec.py --config config/config.yaml --run_id exp1 --max_users 200
   ```

4. **Run Ablations**
   ```bash
   python scripts/step2_run_ablations.py --config config/config.yaml --run_id exp1 --max_users 200
   ```

5. **Evaluate**
   ```bash
   python scripts/step2_evaluate.py --config config/config.yaml --run_id exp1 --methods single_llm,pcnrec,pcnrec_no_verifier,pcnrec_no_negotiation
   ```

6. **Smoke Test Step 2**
   ```bash
   python scripts/smoke_test_step2.py
   ```

## Colab Usage
The scripts are designed to be runnable on Google Colab.
For large runs, ensure you have the `outputs` directory mounted or saved periodically.
For the smoke test, the defaults are small enough for any CPU.
