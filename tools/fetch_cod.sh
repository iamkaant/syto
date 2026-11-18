#!/usr/bin/env bash
# Mirror the Crystallography Open Database CIFs.
#
# COD is CC0 (public domain). Expect tens of GB: a full checkout was 11 GB in
# 2011 with far fewer entries, and there are ~535,000 today. rsync is
# incremental, so re-running only fetches what changed.
#
#   tools/fetch_cod.sh /scratch/cod
#
# Then, as a cluster array job (one task per shard):
#   python tools/cod_torsions.py --cif-dir /scratch/cod/cif \
#       --shard ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT} \
#       --out /scratch/cod/shards
# and when they finish:
#   python tools/cod_torsions.py --merge /scratch/cod/shards \
#       --out data-src/external/
set -euo pipefail
DEST="${1:?usage: fetch_cod.sh <destination-directory>}"
mkdir -p "$DEST/cif"
echo "==> rsyncing COD CIFs into $DEST/cif (tens of GB, resumable)"
rsync -av --delete rsync://www.crystallography.net/cif/ "$DEST/cif/"
echo "==> done. $(find "$DEST/cif" -name '*.cif' | wc -l) CIF files"
