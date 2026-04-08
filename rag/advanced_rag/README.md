# Install all of the dependencies here
python -m pip install -e .

# To run the evaluate_new.py file in Powershell:
python -m rag.advanced_rag.evaluate_new --run-pipeline --force-regenerate --train-samples 250 --test-samples 2000 --json
## If the above does not work, try running the below using Modal
py -m modal run rag/advanced_rag/evaluate_new.py --train-samples 250 --test-samples 2000 --seed 42 --dataset-path /root/kb/gemini_annotated_results.jsonl --generated-output-jsonl /results/metrics/advanced_rag_pipeline_outputs.modal.2250.patched.jsonl --generation-metrics-json /results/metrics/advanced_rag_pipeline_metrics.modal.2250.patched.json --evaluation-metrics-json /results/metrics/reranker.modal.train250.test2000.patched.metrics.json --max-samples 2250

<!-- <ul>
<li>Do note that you will need to have the </li>
</ul> -->

## If you need to enable environment:
.\.venv\Scripts\Activate.ps1

### Do note that the evaluate_outputs.py is the old file with 20 samples as the evaluate_new.py is an improved version of the old file

### Check reranker.modal.train250.test2000.metrics.json for the finalized metrics
### Do switch the model from Qwen/Qwen2.5-0.7B-Instruct Qwen/Qwen2.5-0.5B-Instruct in the
### advanced_rag_pipeline.py for faster generation and basic inference of the model