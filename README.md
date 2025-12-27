### Pull Repository
```shell
git clone https://github.com/toneyavuz/gemma-pytorch-local.git
cd gemma-pytorch-local
```

### Download model files
[https://www.kaggle.com/models/google/gemma-3/pyTorch/gemma-3-4b-it](https://www.kaggle.com/models/google/gemma-3/pyTorch/gemma-3-4b-it)
```
copy gemma-3-4b-it models/gemma-3-4b-it
```

### Clone Gemma PyTorch
```shell
git clone https://github.com/google/gemma_pytorch.git
```

![Folder Structure](/assets/images/folder-structure.png)
### Set Python Environments
```shell
pyenv local 3.12.12
python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install torch==2.6.0
pip install -e ./gemma_pytorch
```

### Check Setup
```shell
python - <<'EOF'
import gemma, torch
print("gemma:", gemma.__file__)
print("torch:", torch.__version__)
EOF
```

### Run Gemma3 model(4b it)
```shell
source .venv/bin/activate
python run_gemma3_4b_it_local.py
```
