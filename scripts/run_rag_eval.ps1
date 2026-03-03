$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Resolve-Path (Join-Path $root "..")
$py = Join-Path $repo "venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$dataset = $env:RAG_EVAL_DATASET
if (-not $dataset) {
  Write-Host "[EVAL] Set RAG_EVAL_DATASET to the dataset path before running."
  exit 1
}

Write-Host "[EVAL] Running RAG evaluation on $dataset ..."
& $py -c "from app.evals.ragas_runner import run_rag_eval; run_rag_eval(r'$dataset')"
