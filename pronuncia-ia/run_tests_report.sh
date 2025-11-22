#!/usr/bin/env bash
set -euo pipefail

# Run unit and integration tests and produce self-contained HTML reports.
# Usage: ./run_tests_report.sh [all|unit|integration]

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPORT_DIR="$ROOT_DIR/tests/reports"
mkdir -p "$REPORT_DIR"

case "${1-}" in
  unit)
    echo "Running unit tests and producing HTML report..."
    pytest -m unit --html="$REPORT_DIR/report_unit.html" --self-contained-html -q
    ;;
  integration)
    echo "Running integration tests and producing HTML report..."
    pytest -m integration --html="$REPORT_DIR/report_integration.html" --self-contained-html -q
    ;;
  all|"" )
    echo "Running unit tests..."
    pytest -m unit --html="$REPORT_DIR/report_unit.html" --self-contained-html -q || true
    echo "Running integration tests..."
    pytest -m integration --html="$REPORT_DIR/report_integration.html" --self-contained-html -q || true
    ;;
  *)
    echo "Usage: $0 [all|unit|integration]"
    exit 2
    ;;
esac

echo "Reports written to: $REPORT_DIR"
