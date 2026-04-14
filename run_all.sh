#!/bin/bash
set -a
source .env
set +a

export PYTHONUNBUFFERED=1

WORKERS=10
VENV=".venv/bin/python"

providers=(
  azure
  deepgram-nova3
  elevenlabs-scribe-v2
  google-chirp3
  openai-gpt4o-transcribe
)

echo "============================================"
echo "Running normalization + scoring for ${#providers[@]} providers"
echo "Concurrency: $WORKERS"
echo "============================================"

for name in "${providers[@]}"; do
  echo ""
  echo "============================================"
  echo "PROVIDER: $name"
  echo "============================================"

  echo "--- Normalizing $name ---"
  $VENV -m scoring.normalize \
    --submission-dir "submissions/raw/$name" \
    --manifest manifest.json \
    --num-workers "$WORKERS" || echo "WARNING: normalize failed for $name, continuing..."

  echo "--- Scoring $name ---"
  $VENV -m scoring.score \
    --submission-dir "submissions/raw/$name" \
    --normalized-dir "submissions/normalized/$name" \
    --manifest manifest.json \
    --output-dir "results/$name" \
    --num-workers "$WORKERS" || echo "WARNING: score failed for $name, continuing..."

  echo "--- Latency stats for $name ---"
  $VENV scripts/latency_stats.py \
    --submission-dir "submissions/raw/$name" \
    --manifest manifest.json \
    --output-dir "results/$name" || echo "WARNING: latency stats failed for $name (no latency.json?), continuing..."

  echo "--- Done with $name ---"
done

echo ""
echo "============================================"
echo "Updating leaderboard..."
echo "============================================"
$VENV -m scoring.update_leaderboard

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
