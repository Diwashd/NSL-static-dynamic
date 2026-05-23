# Reproducing the Python environment

Follow these steps to recreate the development/runtime environment used for the NSL landmark thesis codebase.

1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& "venv\Scripts\Activate.ps1"
```

2. Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) If you plan to run GPU-accelerated TensorFlow, follow TensorFlow's GPU setup docs for your CUDA/cuDNN versions.

4. Run project scripts from the repo root. Example: start the Streamlit app:

```bash
streamlit run app.py
```

Notes:
- The `requirements.txt` contains minimal packages used by the codebase; pip will pull in secondary dependencies automatically.
- If you add new scripts with third-party imports, re-run `pip freeze > requirements.txt` from your environment to capture them.