#!/usr/bin/env bash
# Boot the WebArena docker containers required by the AutoGuard scenarios.
#
# Idempotent: skips load if image is already present, skips start if a
# container with the same name is running.
#
# Usage:
#     ./scripts/start_webarena_containers.sh [pii|news|hack|all]
#
# Defaults to "all". Each scenario uses a single docker container:
#     pii  -> gitlab-populated-final-port8023:latest      (port 8023)
#     news -> postmill-populated-exposed-withimg:latest   (port 9999)
#     hack -> shopping_final_0712:latest                  (port 7770)
#
# Override the image tar location with WEBARENA_IMAGE_DIR (default
# $HOME/webarena-images).
#
# WARNING: Postmill (50G) + Shopping (63G) tarballs together require
# ~115G of free space on the docker storage volume.  Check `df` first.

set -euo pipefail

WEBARENA_IMAGE_DIR="${WEBARENA_IMAGE_DIR:-$HOME/webarena-images}"
SCENARIOS="${1:-all}"

declare -A IMAGE_TAR=(
    [gitlab]="$WEBARENA_IMAGE_DIR/gitlab-populated-final-port8023.tar"
    [postmill]="$WEBARENA_IMAGE_DIR/postmill-populated-exposed-withimg.tar"
    [shopping]="$WEBARENA_IMAGE_DIR/shopping_final_0712.tar"
)
declare -A IMAGE_TAG=(
    [gitlab]="gitlab-populated-final-port8023:latest"
    [postmill]="postmill-populated-exposed-withimg:latest"
    [shopping]="shopping_final_0712:latest"
)
declare -A CONTAINER_NAME=(
    [gitlab]="autoguard-gitlab"
    [postmill]="autoguard-postmill"
    [shopping]="autoguard-shopping"
)
declare -A HOST_PORT=(
    [gitlab]=8023
    [postmill]=9999
    [shopping]=7770
)
declare -A CONTAINER_PORT=(
    [gitlab]=8023
    [postmill]=80
    [shopping]=80
)
# Per-image startup command. GitLab in particular needs an explicit
# runsvdir-start because its default CMD is just `bash` which exits.
declare -A RUN_CMD=(
    [gitlab]="/opt/gitlab/embedded/bin/runsvdir-start"
    [postmill]=""
    [shopping]=""
)
declare -A SHM_SIZE=(
    [gitlab]="10g"
    [postmill]="2g"
    [shopping]="2g"
)

ensure_image() {
    local key=$1
    local tag=${IMAGE_TAG[$key]}
    local tar=${IMAGE_TAR[$key]}

    if docker image inspect "$tag" >/dev/null 2>&1; then
        echo "[image] $tag already present"
        return
    fi
    if [[ ! -f "$tar" ]]; then
        echo "[image] ERROR: tar not found: $tar" >&2
        return 1
    fi
    echo "[image] loading $tar (this may take several minutes)"
    docker load -i "$tar"
}

ensure_running() {
    local key=$1
    local name=${CONTAINER_NAME[$key]}
    local tag=${IMAGE_TAG[$key]}
    local hp=${HOST_PORT[$key]}
    local cp=${CONTAINER_PORT[$key]}

    if docker ps --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "[run] $name already running"
        return
    fi
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "[run] starting existing container $name"
        docker start "$name"
        return
    fi
    local cmd=${RUN_CMD[$key]}
    local shm=${SHM_SIZE[$key]}
    echo "[run] creating container $name from $tag (host:$hp -> container:$cp, shm:$shm)"
    if [[ -n "$cmd" ]]; then
        docker run -d --name "$name" -p "${hp}:${cp}" --shm-size "$shm" "$tag" $cmd
    else
        docker run -d --name "$name" -p "${hp}:${cp}" --shm-size "$shm" "$tag"
    fi
}

case "$SCENARIOS" in
    pii)   keys=(gitlab) ;;
    news)  keys=(postmill) ;;
    hack)  keys=(shopping) ;;
    all)   keys=(gitlab postmill shopping) ;;
    *)
        echo "Usage: $0 [pii|news|hack|all]" >&2
        exit 2
        ;;
esac

for key in "${keys[@]}"; do
    echo
    echo "==> $key"
    ensure_image "$key"
    ensure_running "$key"
done

echo
echo "Containers up. Endpoints:"
for key in "${keys[@]}"; do
    name=${CONTAINER_NAME[$key]}
    hp=${HOST_PORT[$key]}
    case "$key" in
        gitlab)   echo "  PII   GitLab     http://localhost:${hp}    (container: ${name})" ;;
        postmill) echo "  NEWS  Postmill   http://localhost:${hp}    (container: ${name})" ;;
        shopping) echo "  HACK  Shopping   http://localhost:${hp}    (container: ${name})" ;;
    esac
done

echo
echo "Next steps:"
echo "  1. Wait ~30-60s for services to finish starting (especially GitLab)"
echo "  2. (NEWS only) Seed the Trump article:"
echo "       python scripts/seed_postmill_news.py --env-file .env"
echo "  3. Smoke test the URL mapper:"
echo "       python scripts/test_url_mapping.py"
