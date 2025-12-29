# Running on Google Colab Pro

Yes, you can absolutely run this on Colab Pro to speed things up using the new Sharding feature!

## Option A: Faster Local Run (Recommended for Mac M4)
Your ID is running slowly because it's single-threaded. You don't need Colab; you can just run 4 parallel terminals locally:

```bash
# Terminal 1
python scripts/step2_run_pcnrec.py --config config/config.yaml --run_id final_ml100k_w80 --shard 0/4

# Terminal 2
python scripts/step2_run_pcnrec.py --config config/config.yaml --run_id final_ml100k_w80 --shard 1/4

# Terminal 3
python scripts/step2_run_pcnrec.py --config config/config.yaml --run_id final_ml100k_w80 --shard 2/4

# Terminal 4
python scripts/step2_run_pcnrec.py --config config/config.yaml --run_id final_ml100k_w80 --shard 3/4
```
This will cut runtime from 3h to ~45 mins.

## Option B: Moving to Colab

If you prefer Colab:

1. **Zip your project**:
   ```bash
   zip -r project.zip . -x "outputs/*" "venv/*" ".git/*"
   ```
   *Note: Exclude `outputs` to keep it small, but you MUST manually copy `outputs/final_ml100k_w80` structure or regenerate candidates there.*
   
   *Actually, better to zip the essential data inputs too:*
   ```bash
   zip -r project_with_data.zip . -x "venv/*" ".git/*"
   ```

2. **Upload `project_with_data.zip` to Google Drive.**

3. **In Colab Notebook**:
   ```python
   # Mount Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Unzip
   !cp /content/drive/MyDrive/project_with_data.zip .
   !unzip -q project_with_data.zip
   
   # Install
   !pip install -r requirements.txt
   !pip install google-genai
   
   # Run (use Sharding if running multiple notebooks)
   # Set API Key in Secrets or Env
   import os
   os.environ['GEMINI_API_KEY'] = "YOUR_KEY"
   
   !python scripts/step2_run_pcnrec.py --config config/config.yaml --run_id final_ml100k_w80
   ```

4. **Copy results back**:
   ```python
   !zip -r results.zip outputs/final_ml100k_w80
   !cp results.zip /content/drive/MyDrive/
   ```
