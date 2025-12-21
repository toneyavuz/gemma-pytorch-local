```shell
pyenv local 3.12.12
python -m venv .venv
source .venv/bin/activate

pip install -U pip setuptools wheel
pip install torch==2.6.0
pip install -e ./gemma_pytorch
```

```shell
python - <<'EOF'
import gemma, torch
print("gemma:", gemma.__file__)
print("torch:", torch.__version__)
EOF
```

```shell
source .venv/bin/activate
python run_gemma3_4b_it_local.py
```
