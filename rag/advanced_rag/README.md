# To run the evaluate_new.py file in Powershell:
python -m rag.advanced_rag.evaluate_new --run-pipeline --force-regenerate --train-samples 250 --test-samples 2000 --json
## If you need to enable environment:
.\.venv\Scripts\Activate.ps1

### Do note that the evaluate_outputs.py is the old file with 20 samples as the evaluate_new.py is an improved version of the old file