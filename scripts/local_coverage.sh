#!/bin/bash
# Run full test suite with coverage locally and optionally upload to Codecov
# Usage: ./scripts/local_coverage.sh [--upload]

set -e

cd "$(dirname "$0")/.."

echo "========================================="
echo "Full Test Suite with Coverage"
echo "========================================="
echo ""
echo "Running ALL tests (including MLX tests)..."
echo "This may take 15-30 seconds."
echo ""

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run full unit tests with coverage
pytest -m unit \
    --cov=src \
    --cov-report=xml \
    --cov-report=html \
    --cov-report=term-missing \
    -v

echo ""
echo "========================================="
echo "Coverage Report Generated"
echo "========================================="
echo ""
echo "View HTML report: open htmlcov/index.html"
echo "XML report saved: coverage.xml"
echo ""

# Check coverage thresholds
echo "Checking module-specific coverage..."
python - <<'EOF'
import xml.etree.ElementTree as ET
import sys

tree = ET.parse('coverage.xml')
root = tree.getroot()

# Module-specific coverage requirements (just basenames to match coverage.xml format)
critical_modules = {
    'distrust_loss.py': 90,
    'citation_scorer.py': 85,
    'metrics.py': 85,
    'config.py': 80,
}

print("\nModule Coverage:")
print("-" * 60)

found_modules = set()
failed = []

for package in root.findall('.//package'):
    for cls in package.findall('.//class'):
        filename = cls.get('filename')
        line_rate = float(cls.get('line-rate', 0)) * 100

        if filename in critical_modules:
            found_modules.add(filename)
            threshold = critical_modules[filename]
            status = "✓" if line_rate >= threshold else "❌"
            print(f"{status} {filename}: {line_rate:.1f}% (target: {threshold}%)")

            if line_rate < threshold:
                failed.append(filename)

print("-" * 60)

# Verify all critical modules were found
missing_modules = set(critical_modules.keys()) - found_modules
if missing_modules:
    print("\n❌ ERROR: Critical modules not found in coverage report:")
    for module in missing_modules:
        print(f"  - {module}")
    print("\nThis likely means the coverage report format changed or modules were not imported.")
    sys.exit(1)

if failed:
    print(f"\n❌ WARNING: {len(failed)} module(s) below threshold")
    sys.exit(1)

print(f"\n✓ All {len(found_modules)} critical modules meet coverage thresholds")
EOF

# Optional upload to Codecov
if [ "$1" == "--upload" ]; then
    echo ""
    echo "Uploading to Codecov..."

    if [ -z "$CODECOV_TOKEN" ]; then
        echo "ERROR: CODECOV_TOKEN environment variable not set"
        echo "Export it first: export CODECOV_TOKEN=your_token"
        exit 1
    fi

    # Upload with full-suite flag
    curl -Os https://uploader.codecov.io/latest/macos/codecov
    chmod +x codecov
    ./codecov -t $CODECOV_TOKEN -f coverage.xml -F full-suite-local
    rm codecov

    echo "✓ Coverage uploaded to Codecov"
fi

echo ""
echo "Done! Open htmlcov/index.html to view detailed coverage."

