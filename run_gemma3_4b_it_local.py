import os
import torch
import contextlib

from gemma.config import get_model_config
from gemma.gemma3_model import Gemma3ForMultimodalLM


# -------------------------------------------------
# PATH (script neredeyse oradan çözer)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "gemma-3-4b-it")

CKPT_PATH = os.path.join(MODEL_DIR, "model.ckpt")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.model")

assert os.path.isfile(CKPT_PATH), f"model.ckpt yok: {CKPT_PATH}"
assert os.path.isfile(TOKENIZER_PATH), f"tokenizer.model yok: {TOKENIZER_PATH}"


# -------------------------------------------------
# Device
# -------------------------------------------------
DEVICE = torch.device("cpu")
DTYPE = torch.float32


@contextlib.contextmanager
def set_default_dtype(dtype):
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def build_prompt(text: str) -> str:
    return (
        "<start_of_turn>user\n"
        f"{text}"
        "<end_of_turn><eos>\n"
        "<start_of_turn>model\n"
    )


def is_gemma3_arch(cfg) -> bool:
    """Repo sürümlerine göre enum/string farkı olabiliyor; güvenli kontrol."""
    arch = getattr(cfg, "architecture", None)
    if arch is None:
        return False
    name = getattr(arch, "name", None)
    if isinstance(name, str):
        return name == "GEMMA_3"
    s = str(arch)
    return ("GEMMA_3" in s) or s.endswith("GEMMA_3")


def pick_variant_for_gemma3() -> str:
    # Senin repoda desteklenenler bunlar (hata mesajından)
    candidates = ["27b_v3", "27b", "12b", "9b", "7b", "4b", "2b-v2", "2b", "1b"]

    good = []
    for v in candidates:
        try:
            cfg = get_model_config(v)
        except Exception:
            continue
        if is_gemma3_arch(cfg):
            good.append(v)

    if not good:
        raise RuntimeError(
            "Bu gemma_pytorch sürümünde GEMMA_3 mimarili config bulunamadı.\n"
            "Muhtemelen yanlış repo sürümü / yanlış model sınıfı kullanıyorsun."
        )

    # Öncelik 4b
    if "4b" in good:
        return "4b"
    # Öncelik 27b_v3
    if "27b_v3" in good:
        return "27b_v3"
    return good[0]


def main():
    print(f"Device: {DEVICE} | dtype: {DTYPE}")

    variant = pick_variant_for_gemma3()
    print("Selected variant:", variant)

    cfg = get_model_config(variant)
    cfg.tokenizer = TOKENIZER_PATH
    cfg.dtype = "float32"

    # Ek debug: mimari ne?
    print("Config architecture:", getattr(cfg, "architecture", None))

    print("Instantiating Gemma3ForMultimodalLM...")
    with set_default_dtype(DTYPE):
        model = Gemma3ForMultimodalLM(cfg)

    print("Loading checkpoint to CPU...")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Missing keys: {len(missing)}")
    print(f"Unexpected keys: {len(unexpected)}")

    model.eval()
    print("\nModel loaded ✅")

    # CPU'da generate çok yavaş olabilir; küçük tut
    prompt = build_prompt("Merhaba! nasilsin?")
    print("\nPROMPT:\n", prompt)

    with torch.no_grad():
        outs = model.generate(
            [prompt],          # <<< ÖNEMLİ: string değil, liste!
            device=DEVICE,
            output_len=64,
        )

    print("\nOUTPUT:\n", outs)

if __name__ == "__main__":
    main()
