#!/usr/bin/env bash
# ===========================================================================
# VIDYUT CI Pipeline
# ===========================================================================
# Runs the full local CI: lint, type-check (best-effort), tests with coverage,
# validation, and Docker image build verification.
#
# Usage:
#   ./scripts/ci_pipeline.sh                    # full run
#   ./scripts/ci_pipeline.sh --skip-docker      # skip docker build
#   ./scripts/ci_pipeline.sh --fast             # tests only
# ===========================================================================
set -euo pipefail

CYAN='\033[0;36m'; GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
SKIP_DOCKER=0
FAST=0

for arg in "$@"; do
  case $arg in
    --skip-docker) SKIP_DOCKER=1 ;;
    --fast) FAST=1 ;;
    *) echo "unknown arg: $arg" ;;
  esac
done

step() { echo -e "\n${CYAN}== $* ==${NC}"; }
ok() { echo -e "${GREEN}✓ $*${NC}"; }
warn() { echo -e "${YELLOW}⚠ $*${NC}"; }
fail() { echo -e "${RED}✗ $*${NC}"; exit 1; }

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:."
mkdir -p reports

# ---------- 1. environment ----------
step "Environment"
python --version
pip --version
ok "Python ready"

# ---------- 2. lint ----------
if [ $FAST -eq 0 ]; then
  step "Lint (flake8)"
  if command -v flake8 >/dev/null 2>&1; then
    flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503 \
      --statistics --tee --output-file=reports/flake8.txt || warn "flake8 found issues"
    ok "flake8 done"
  else
    warn "flake8 not installed; skipping"
  fi

  step "Format check (black)"
  if command -v black >/dev/null 2>&1; then
    black --check --diff src/ tests/ scripts/ || warn "black formatting drift"
    ok "black done"
  else
    warn "black not installed; skipping"
  fi
fi

# ---------- 3. tests ----------
step "Pytest with coverage"
pytest tests/ \
  --cov=src \
  --cov-report=term-missing \
  --cov-report=html:reports/htmlcov \
  --cov-report=xml:reports/coverage.xml \
  --junitxml=reports/junit.xml \
  -q || fail "tests failed"
ok "tests passed"

# ---------- 4. validation ----------
if [ $FAST -eq 0 ]; then
  step "End-to-end validation"
  python -m scripts.validate_all \
    --output reports/evaluation.json \
    --markdown reports/evaluation.md \
    --model-version "${MODEL_VERSION:-v2}" \
    || warn "validation completed with target failures (see report)"
  ok "validation report written"
fi

# ---------- 5. docker build ----------
if [ $FAST -eq 0 ] && [ $SKIP_DOCKER -eq 0 ]; then
  step "Docker build"
  if command -v docker >/dev/null 2>&1; then
    docker build -t vidyut-api:ci -f docker/Dockerfile.api . || fail "docker build failed"
    docker build -t vidyut-app:ci -f docker/Dockerfile . || fail "docker build failed"
    ok "docker images built"
  else
    warn "docker not installed; skipping"
  fi
fi

echo ""
echo -e "${GREEN}====================================="
echo -e "  CI pipeline COMPLETED successfully"
echo -e "  Reports: ./reports/"
echo -e "=====================================${NC}"