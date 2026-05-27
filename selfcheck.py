import os, sys, py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

BAD_NAMES = {"con","prn","aux","nul","com1","com2","com3","com4","com5","com6","com7","com8","com9","lpt1","lpt2","lpt3","lpt4","lpt5","lpt6","lpt7","lpt8","lpt9"}

def main():
    # 1) compile all py
    py_files = list(ROOT.rglob("*.py"))
    for p in py_files:
        py_compile.compile(str(p), doraise=True)
    # 2) check reserved names at top-level
    for p in ROOT.iterdir():
        if p.name.lower() in BAD_NAMES:
            raise SystemExit(f"[FAIL] Reserved Windows device name in package: {p.name}")
    # 3) ensure core/trainer has trainer_engine.train
    import importlib.util
    spec = importlib.util.spec_from_file_location("core.trainer", ROOT/"core"/"trainer.py")
    mod = importlib.util.module_from_spec(spec)
    # dataclass 等依賴 sys.modules[__name__] 存在；先註冊再載入。
    sys.modules[spec.name] = mod  # type: ignore
    spec.loader.exec_module(mod)  # type: ignore
    te = getattr(mod, "trainer_engine", None)
    if te is None or not hasattr(te, "train"):
        raise SystemExit("[FAIL] trainer_engine.train is missing")
    print("[OK] Selfcheck passed. Python files compile and core interface is present.")

if __name__ == "__main__":
    main()
