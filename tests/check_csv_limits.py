"""Run from the repository root after make: python3 tests/check_csv_limits.py."""
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRAM = ROOT / "ml_core"
# Each entry: menu option, algorithm-specific keyboard input.
CASES = [(1, "7"), (2, "2"), (3, "2 2"), (4, "2"), (5, ""), (6, "3\n2 2")]


def rows(option, count):
    result = []
    for i in range(count):
        x = i + 1
        y = 2 * x + (i % 3)
        if option == 3:
            label = -1 if i % 2 == 0 else 1
            result.append(f"{label * x},{label * y},{label}")
        elif option == 6:
            result.append(f"{x},{y},{i % 2}")
        else:
            result.append(f"{x},{y}")
    return "\n".join(result) + "\n"


def run(option, path, keyboard):
    inputs = f"{option}\n{path}\n{keyboard}\n0\n"
    result = subprocess.run(
        [str(PROGRAM)], input=inputs, text=True,
        capture_output=True, cwd=ROOT, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


with tempfile.TemporaryDirectory(prefix="mlcore-csv-") as folder:
    for option, keyboard in CASES:
        path = Path(folder) / "data with spaces.csv"
        for count in (99, 100):
            path.write_text(rows(option, count))
            output = run(option, path, keyboard)
            assert "could not complete" not in output, output
        path.write_text(rows(option, 101))
        # Rejected files require no algorithm-specific keyboard input.
        output = run(option, path, "")
        assert "Maximum 100 data rows allowed" in output, output
        for malformed in ("1,\n2\n3,6\n", "1,2,3,4\n", ""):
            path.write_text(malformed)
            output = run(option, path, "")
            assert "could not complete" in output, output
        print(f"Option {option}: 99/100 rows accepted; 101 and malformed files rejected.")
print("CSV checks passed.")
