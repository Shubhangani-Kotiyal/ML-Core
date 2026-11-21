"""Run after make: python3 tests/check_kmeans.py."""
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(points, k):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'points.csv'
        path.write_text(''.join(f'{x},{y}\n' for x, y in points))
        result = subprocess.run(
            [str(ROOT / 'ml_core')], input=f'2\n{path}\n{k}\n0\n',
            text=True, capture_output=True, cwd=ROOT, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout


def check_clusters(points, k):
    output = run(points, k)
    assert 'could not complete' not in output, output
    groups = re.findall(
        r'Cluster \d+\nCentroid: \([^\n]+\)\n((?:\([^\n]+\)\n)*)', output)
    assert len(groups) == k, output
    assert all(group.strip() for group in groups), output
    assert sum(len(group.splitlines()) for group in groups) == len(points), output
    assert 'Warning:' not in output, output
    return output


repeated = [(0, 0), (0, 0), (10, 10), (10, 10)]
assert 'K cannot exceed the number of distinct points' in run(repeated, 3)
check_clusters(repeated, 2)
check_clusters([(1, 1)] * 4, 1)
assert 'K cannot exceed the number of distinct points' in run([(1, 1)] * 4, 2)
# Distinct initial centers, but a cluster becomes empty after reassignment.
check_clusters([(3, 0), (7, 6), (7, 5), (2, 5), (3, 0),
                (5, 6), (0, 5), (5, 7), (0, 6), (1, 4)], 3)
sample = [(1, 1), (1.5, 2), (2, 1.5), (2.5, 2),
          (5, 5), (5.5, 6), (6, 5.5), (6.5, 6)]
output = check_clusters(sample, 2)
assert 'Centroid: (1.75, 1.625)' in output, output
assert 'Centroid: (5.75, 5.625)' in output, output
print('K-Means duplicate, empty-cluster, and sample checks passed.')
