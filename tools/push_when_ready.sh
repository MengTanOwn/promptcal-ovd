#!/usr/bin/env bash
# Push the repository to GitHub once the deploy key has been authorised.
#
# github.com:443 is not reachable from this machine (the local proxy answers 502
# for the CONNECT and a direct connection times out), while ssh.github.com:443 is
# reachable directly.  ~/.ssh/config therefore maps github.com to ssh.github.com:443,
# and this script waits until the public key is registered, then pushes.
#
#   bash tools/push_when_ready.sh          # wait up to 20 minutes
#   MAX_TRIES=2 bash tools/push_when_ready.sh
set -u

MAX_TRIES="${MAX_TRIES:-80}"
SLEEP="${SLEEP:-15}"
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new"
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY

for i in $(seq 1 "$MAX_TRIES"); do
    if ssh -o BatchMode=yes -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
        echo "key authorised after $((i * SLEEP))s; pushing"
        git push -u origin main
        echo "PUSH_EXIT=$?"
        exit 0
    fi
    sleep "$SLEEP"
done

echo "TIMEOUT: key was never authorised"
exit 1
