#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="${PROJECT_ROOT}/cuda_version.txt"
DEVICE_QUERY_DIR="${PROJECT_ROOT}/deviceQuery"
DEVICE_QUERY_BINARY="${DEVICE_QUERY_DIR}/deviceQuery"
BUILD_LOG="${DEVICE_QUERY_DIR}/cuda_build_log.txt"
DEFAULT_CCAP="${DEFAULT_CCAP:-89}"
LOG_PREFIX="[detect_cuda]"

log() {
    printf '%s %s\n' "${LOG_PREFIX}" "$1" >&2
}

sanitize_ccap() {
    local raw="$1"
    raw="${raw//[^0-9.]/}"
    raw="${raw#.}"
    raw="${raw%.}"
    [[ -z "$raw" ]] && return 1

    local major minor
    if [[ "$raw" == *.* ]]; then
        major="${raw%%.*}"
        minor="${raw##*.}"
    else
        major="$raw"
        minor="0"
    fi

    [[ -z "$major" || -z "$minor" ]] && return 1
    [[ "$major" =~ ^[0-9]+$ ]] || return 1
    [[ "$minor" =~ ^[0-9]+$ ]] || return 1

    major=$((10#$major))
    minor=$((10#$minor))
    printf '%d\n' $((major * 10 + minor))
}

pick_best_ccap() {
    local best=""
    for candidate in "$@"; do
        [[ -z "$candidate" ]] && continue
        if [[ -z "$best" || "$candidate" -gt "$best" ]]; then
            best="$candidate"
        fi
    done
    [[ -n "$best" ]] || return 1
    printf '%s\n' "$best"
}

parse_visible_devices() {
    local visible="${CUDA_VISIBLE_DEVICES:-}"
    visible="${visible// /}"
    visible="${visible%%,*}"
    if [[ -n "$visible" && "$visible" != "NoDevFiles" ]]; then
        printf '%s\n' "$visible"
    fi
}

from_nvidia_smi() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1

    local -a rows=()
    mapfile -t rows < <(nvidia-smi --query-gpu=index,compute_cap --format=csv,noheader 2>/dev/null || true)
    [[ ${#rows[@]} -gt 0 ]] || return 1

    local visible_index=""
    visible_index=$(parse_visible_devices || true)

    local -A index_to_cap=()
    local row idx cap
    for row in "${rows[@]}"; do
        IFS=',' read -r idx cap <<<"$row"
        idx="${idx//[!0-9]/}"
        cap="${cap// /}"
        [[ -z "$idx" || -z "$cap" || "$cap" == "N/A" ]] && continue
        local normalized
        normalized=$(sanitize_ccap "$cap" || true)
        [[ -n "$normalized" ]] || continue
        index_to_cap[$idx]="$normalized"
    done

    if [[ -n "$visible_index" && -n "${index_to_cap[$visible_index]:-}" ]]; then
        printf '%s\n' "${index_to_cap[$visible_index]}"
        log "nvidia-smi compute capability sm_${index_to_cap[$visible_index]} (CUDA_VISIBLE_DEVICES=${visible_index})"
        return 0
    fi

    if [[ ${#index_to_cap[@]} -eq 0 ]]; then
        return 1
    fi

    local -a caps=()
    for cap in "${index_to_cap[@]}"; do
        caps+=("$cap")
    done

    local best
    best=$(pick_best_ccap "${caps[@]}") || return 1
    log "nvidia-smi compute capability sm_${best}"
    printf '%s\n' "$best"
}

from_gpu_name() {
    command -v nvidia-smi >/dev/null 2>&1 || return 1

    declare -A name_map=(
        ["NVIDIA GeForce RTX 4090"]="89"
        ["NVIDIA GeForce RTX 4090 D"]="89"
        ["NVIDIA GeForce RTX 4090D"]="89"
        ["NVIDIA GeForce RTX 4080"]="89"
        ["NVIDIA H100"]="90"
        ["NVIDIA H200"]="90"
        ["NVIDIA B100"]="120"
        ["NVIDIA B200"]="120"
        ["NVIDIA Blackwell B100"]="120"
        ["NVIDIA Blackwell B200"]="120"
        ["NVIDIA GeForce RTX 5090"]="100"
        ["NVIDIA GeForce RTX 5090 Ti"]="100"
        ["NVIDIA GeForce RTX 5080"]="100"
    )

    local -a rows=()
    mapfile -t rows < <(nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true)
    [[ ${#rows[@]} -gt 0 ]] || return 1

    local visible_index
    visible_index=$(parse_visible_devices || true)

    local chosen=""
    local row idx name
    for row in "${rows[@]}"; do
        IFS=',' read -r idx name <<<"$row"
        idx="${idx//[!0-9]/}"
        name="${name## }"
        name="${name%% }"
        [[ -z "$idx" || -z "$name" ]] && continue
        local mapped="${name_map[$name]:-}"
        if [[ -z "$mapped" ]]; then
            if [[ "$name" =~ RTX[[:space:]]50[0-9]{2} ]]; then
                mapped="100"
            elif [[ "$name" =~ H[12]00 ]]; then
                mapped="90"
            elif [[ "$name" =~ B[12]00 ]]; then
                mapped="120"
            fi
        fi
        [[ -z "$mapped" ]] && continue
        if [[ -n "$visible_index" && "$idx" == "$visible_index" ]]; then
            chosen="$mapped"
            break
        fi
        if [[ -z "$chosen" || "$mapped" -gt "$chosen" ]]; then
            chosen="$mapped"
        fi
    done

    [[ -n "$chosen" ]] || return 1
    log "nvidia-smi gpu name map compute capability sm_${chosen}"
    printf '%s\n' "$chosen"
}

build_device_query() {
    command -v nvcc >/dev/null 2>&1 || return 1
    [[ -d "$DEVICE_QUERY_DIR" ]] || return 1
    if [[ -x "$DEVICE_QUERY_BINARY" ]]; then
        return 0
    fi
    log "Building NVIDIA deviceQuery sample (output in ${BUILD_LOG})"
    if ! make -C "$DEVICE_QUERY_DIR" deviceQuery >"$BUILD_LOG" 2>&1; then
        log "Failed to build deviceQuery (see ${BUILD_LOG})"
        return 1
    fi
    return 0
}

from_device_query() {
    build_device_query || return 1
    local capability_line
    capability_line=$("$DEVICE_QUERY_BINARY" 2>/dev/null | awk -F ':' '/CUDA Capability/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' || true)
    [[ -n "$capability_line" ]] || return 1
    local normalized
    normalized=$(sanitize_ccap "$capability_line" || true)
    [[ -n "$normalized" ]] || return 1
    log "deviceQuery compute capability sm_${normalized}"
    printf '%s\n' "$normalized"
}

main() {
    local ccap=""

    ccap=$(from_nvidia_smi || true)
    if [[ -z "$ccap" ]]; then
        ccap=$(from_gpu_name || true)
    fi
    if [[ -z "$ccap" ]]; then
        ccap=$(from_device_query || true)
    fi
    if [[ -z "$ccap" ]]; then
        ccap="$DEFAULT_CCAP"
        log "Autodetection failed, using fallback sm_${ccap}"
    fi

    printf '%s\n' "$ccap" >"$OUTPUT_FILE"
    log "Wrote sm_${ccap} to $(basename "$OUTPUT_FILE")"
    printf '%s\n' "$ccap"
}

main "$@"
