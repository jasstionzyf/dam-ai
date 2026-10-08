#!/bin/bash
# Build the dam-ai tagger engine image from the repo root context.
set -euo pipefail
cd "$(dirname "$0")/../.."
docker build -f deploy/tagger/Dockerfile -t damai-tagger:v0.1 .
