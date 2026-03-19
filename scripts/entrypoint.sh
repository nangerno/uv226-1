#!/bin/bash
set -e
redis-server --daemonize yes
sleep 2  # Give Redis a moment to start up
echo "*****Running text trainer*****"
# Avoid numexpr "NUMEXPR_MAX_THREADS not set" log when many cores are present
export NUMEXPR_MAX_THREADS="${NUMEXPR_MAX_THREADS:-16}"
# Reduce CUDA fragmentation (helps with OOM when reserved but unallocated memory is large)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Silence "df: /root/.triton/autotune: No such file or directory" from Triton
mkdir -p /root/.triton/autotune 2>/dev/null || true
python -m text_trainer "$@"