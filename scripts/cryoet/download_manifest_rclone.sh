#!/usr/bin/env bash
set -euo pipefail

MANIFEST_TSV="${1:?usage: download_manifest_rclone.sh <download_manifest.tsv>}"
DOWNLOAD_JOBS="${DOWNLOAD_JOBS:-4}"
LOG_DIR="${LOG_DIR:-$(dirname "$MANIFEST_TSV")/logs/download_rclone}"
RCLONE_RETRIES="${RCLONE_RETRIES:-10}"
RCLONE_LOW_LEVEL_RETRIES="${RCLONE_LOW_LEVEL_RETRIES:-20}"
RCLONE_REMOTE_PREFIX="${RCLONE_REMOTE_PREFIX:-:s3,provider=AWS,env_auth=false,anonymous=true:cryoet-data-portal-public}"

mkdir -p "$LOG_DIR"

download_one() {
    local run_name="$1"
    local url="$2"
    local output_path="$3"
    local expected_size="$4"
    local remote_path="${url#https://files.cryoetdataportal.cziscience.com/}"
    local remote="${RCLONE_REMOTE_PREFIX}/${remote_path}"
    local tmp_path="${output_path}.rclone"
    local log_path="${LOG_DIR}/${run_name}.log"

    mkdir -p "$(dirname "$output_path")"

    if [[ -f "$output_path" ]]; then
        local actual_size
        actual_size="$(stat -c '%s' "$output_path")"
        if [[ "$actual_size" == "$expected_size" ]]; then
            echo "SKIP ${run_name} size=${actual_size}"
            return 0
        fi
        rm -f "$output_path"
    fi

    rm -f "${output_path}.part" "$tmp_path"

    rclone copyto \
        --s3-no-check-bucket \
        --checkers=1 \
        --transfers=1 \
        --multi-thread-streams=4 \
        --retries="$RCLONE_RETRIES" \
        --low-level-retries="$RCLONE_LOW_LEVEL_RETRIES" \
        --log-file="$log_path" \
        --log-level=INFO \
        "$remote" \
        "$tmp_path"

    local tmp_size
    tmp_size="$(stat -c '%s' "$tmp_path")"
    if [[ "$tmp_size" != "$expected_size" ]]; then
        echo "ERROR ${run_name} expected=${expected_size} actual=${tmp_size}" >&2
        return 1
    fi

    mv -f "$tmp_path" "$output_path"
    echo "DONE ${run_name} size=${tmp_size}"
}

status=0
active=0

while IFS=$'\t' read -r run_name url output_path expected_size; do
    if [[ "$run_name" == "run_name" ]]; then
        continue
    fi

    download_one "$run_name" "$url" "$output_path" "$expected_size" &
    ((active += 1))

    if (( active >= DOWNLOAD_JOBS )); then
        if ! wait -n; then
            status=1
        fi
        active="$(jobs -pr | wc -l)"
    fi
done < "$MANIFEST_TSV"

while (( $(jobs -pr | wc -l) > 0 )); do
    if ! wait -n; then
        status=1
    fi
done

exit "$status"
