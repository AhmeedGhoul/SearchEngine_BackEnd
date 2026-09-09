Write-Host "Creating .venv_asr..."
python -m venv .venv_asr

Write-Host "Installing ASR dependencies..."
.\.venv_asr\Scripts\pip.exe install -r requirements-asr.txt

Write-Host "Done. Test with:"
Write-Host "  .\.venv_asr\Scripts\python.exe -c `"import torch; print('CUDA:', torch.cuda.is_available())`""
