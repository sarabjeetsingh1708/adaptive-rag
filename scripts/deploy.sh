#!/bin/bash
# scripts/deploy.sh
# One-command deploy to Fly.io.
# Run this from the project root after setting up your .env

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}▶ Adaptive RAG — Deploy to Fly.io${NC}\n"

# ── Pre-flight checks ──────────────────────────────────────────────────────
echo "Checking prerequisites..."

if ! command -v fly &> /dev/null; then
  echo -e "${RED}✗ Fly CLI not installed.${NC}"
  echo "  Install: curl -L https://fly.io/install.sh | sh"
  exit 1
fi
echo -e "  ${GREEN}✓ fly CLI found${NC}"

if ! fly auth whoami &> /dev/null; then
  echo -e "${RED}✗ Not logged in to Fly.io${NC}"
  echo "  Run: fly auth login"
  exit 1
fi
echo -e "  ${GREEN}✓ Fly.io authenticated${NC}"

if [ ! -f .env ]; then
  echo -e "${RED}✗ .env file not found${NC}"
  echo "  Run: cp .env.example .env && add your GEMINI_API_KEY"
  exit 1
fi

GEMINI_KEY=$(grep GEMINI_API_KEY .env | cut -d'=' -f2)
if [ -z "$GEMINI_KEY" ] || [ "$GEMINI_KEY" = "your_gemini_api_key_here" ]; then
  echo -e "${RED}✗ GEMINI_API_KEY not set in .env${NC}"
  exit 1
fi
echo -e "  ${GREEN}✓ GEMINI_API_KEY found${NC}"

VECTORS=$(python -c "
import sys; sys.path.insert(0, '.')
from src.retrieval.vector_store import VectorStore
try:
  print(VectorStore().count())
except:
  print(0)
" 2>/dev/null || echo 0)

if [ "$VECTORS" -eq "0" ]; then
  echo -e "${YELLOW}⚠ Vector store is empty. Have you run scripts/ingest.py?${NC}"
  read -p "  Continue anyway? [y/N] " yn
  [[ "$yn" == "y" ]] || exit 1
else
  echo -e "  ${GREEN}✓ Vector store: $VECTORS vectors indexed${NC}"
fi

echo ""

# ── First deploy vs update ─────────────────────────────────────────────────
if ! fly status &> /dev/null; then
  echo "First deploy — running fly launch..."
  fly launch --no-deploy --name adaptive-rag --region sin
fi

# ── Set secrets ───────────────────────────────────────────────────────────
echo "Setting Fly secrets..."
fly secrets set GEMINI_API_KEY="$GEMINI_KEY"

TAVILY_KEY=$(grep TAVILY_API_KEY .env | cut -d'=' -f2)
if [ -n "$TAVILY_KEY" ] && [ "$TAVILY_KEY" != "your_tavily_api_key_here" ]; then
  fly secrets set TAVILY_API_KEY="$TAVILY_KEY"
  echo -e "  ${GREEN}✓ TAVILY_API_KEY set${NC}"
fi

# ── Create persistent volume for Qdrant data ──────────────────────────────
if ! fly volumes list | grep -q qdrant_data; then
  echo "Creating persistent volume for Qdrant..."
  fly volumes create qdrant_data --size 1 --region sin
fi

# ── Deploy ─────────────────────────────────────────────────────────────────
echo ""
echo "Deploying..."
fly deploy --remote-only

echo ""
echo -e "${GREEN}✓ Deployed!${NC}"
URL=$(fly status --json | python3 -c "import sys,json; d=json.load(sys.stdin); print('https://' + d.get('Hostname','adaptive-rag.fly.dev'))" 2>/dev/null || echo "https://adaptive-rag.fly.dev")
echo -e "  Live at: ${GREEN}$URL${NC}"
echo ""
echo "Next steps:"
echo "  1. Open $URL — paste this URL in every job application"
echo "  2. Run ingest on the deployed machine:"
echo "     fly ssh console -C 'python scripts/ingest.py --papers 500'"
echo "  3. Share on LinkedIn with a screenshot of the eval dashboard"
EOF
chmod +x /home/claude/adaptive-rag/scripts/deploy.sh 2>/dev/null || true
