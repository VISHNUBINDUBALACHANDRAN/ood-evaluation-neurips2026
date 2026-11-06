#!/bin/bash
cd "${REPO_ROOT}"
exec .venv/bin/python run_mcdropout_mechanism.py --arch layoutlmv3 \
  --seeds 42 123 456 789 1024 --rates_doc 0.0 0.1 \
  --n_eval 1000 --n_passes 25 --batch_size_doc 16 \
  --out outputs/mcdropout_mechanism/mcdropout_layoutlm.json
