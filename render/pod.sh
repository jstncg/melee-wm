#!/usr/bin/env bash
# ssh helper. POD=cpu (default) or POD=gpu.   bash render/pod.sh '<cmd>'   |   bash render/pod.sh --scp src dst
U=root
case "${POD:-cpu}" in
  bench) HOST=ssh.runpod.io; PORT=22; U=sn00gye6az3o2b-6441214d ;;
  cpu) HOST=103.196.86.91; PORT=45600 ;;
  gpu) HOST=213.181.111.2; PORT=54281 ;;
  gpu2) HOST=103.196.86.137; PORT=15269 ;;
  gpu3) HOST=47.47.180.26; PORT=19708 ;;
esac
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=20 -i $HOME/.ssh/id_ed25519"
if [ "$1" = "--scp" ]; then shift; exec scp $OPTS -P $PORT "$@"; fi
if [ "$1" = "--host" ]; then echo "$U@$HOST"; exit 0; fi
[ "${POD:-}" = bench ] && OPTS="$OPTS -tt"
ssh $OPTS -p $PORT $U@$HOST "$@"
