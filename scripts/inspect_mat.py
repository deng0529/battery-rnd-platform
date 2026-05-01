from pathlib import Path
from scipy.io import loadmat
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
print(BASE_DIR)
RAW_DIR = BASE_DIR / "data" / "raw"


def describe(obj, name="", depth=0, max_depth=4):
    indent = "  " * depth

    if depth > max_depth:
        return

    print(f"{indent}{name}: {type(obj)}")

    if isinstance(obj, np.ndarray):
        print(f"{indent}  shape={obj.shape}, dtype={obj.dtype}")
        if obj.size == 1:
            describe(obj.item(), name + ".item()", depth + 1, max_depth)

    elif hasattr(obj, "__dict__"):
        fields = [k for k in obj.__dict__.keys() if not k.startswith("_")]
        print(f"{indent}  fields={fields}")

        for field in fields:
            try:
                describe(getattr(obj, field), field, depth + 1, max_depth)
            except Exception as e:
                print(f"{indent}  {field}: ERROR {e}")

def main():
    mat_files = sorted(RAW_DIR.glob("*.mat"))

    if not mat_files:
        print("No .mat files found in data/raw/")
        return

    mat_path = mat_files[0]
    print(f"Inspecting: {mat_path}")

    mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    print("\nTop-level keys:")
    print([k for k in mat.keys() if not k.startswith("__")])

    cell_id = mat_path.stem
    battery = mat[cell_id]

    print("\nBattery structure:")
    describe(battery, cell_id)

    cycles = battery.cycle
    print("\nCycle count:", len(cycles))

    print("\nFirst cycle:")
    describe(cycles[0], "cycle[0]")

    print("\nFirst 10 cycle types:")
    for i, c in enumerate(cycles[:10]):
        print(i + 1, c.type)

if __name__ == "__main__":
    main()