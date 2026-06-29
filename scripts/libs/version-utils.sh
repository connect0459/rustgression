#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

error()   { echo -e "${RED}$1${NC}"; }
warning() { echo -e "${YELLOW}$1${NC}"; }
success() { echo -e "${GREEN}$1${NC}"; }
info()    { echo -e "${BLUE}$1${NC}"; }

to_python_version() {
    local v=$1
    v=${v/-alpha./a}
    v=${v/-beta./b}
    v=${v/-rc./rc}
    echo "$v"
}

compare_versions() {
    local v1=$1
    local v2=$2

    local v1_base=${v1%%-*}
    local v2_base=${v2%%-*}
    local v1_pre=${v1#"$v1_base"}; v1_pre=${v1_pre#-}
    local v2_pre=${v2#"$v2_base"}; v2_pre=${v2_pre#-}

    IFS='.' read -r v1_major v1_minor v1_patch <<< "$v1_base"
    IFS='.' read -r v2_major v2_minor v2_patch <<< "$v2_base"

    if [ "$v1_major" -gt "$v2_major" ]; then return 0; fi
    if [ "$v1_major" -lt "$v2_major" ]; then return 1; fi
    if [ "$v1_minor" -gt "$v2_minor" ]; then return 0; fi
    if [ "$v1_minor" -lt "$v2_minor" ]; then return 1; fi
    if [ "$v1_patch" -gt "$v2_patch" ]; then return 0; fi
    if [ "$v1_patch" -lt "$v2_patch" ]; then return 1; fi

    if [ -z "$v1_pre" ] && [ -n "$v2_pre" ]; then return 0; fi
    if [ -n "$v1_pre" ] && [ -z "$v2_pre" ]; then return 1; fi

    if [ -z "$v1_pre" ] && [ -z "$v2_pre" ]; then return 0; fi

    local v1_pre_type=${v1_pre%%.*}
    local v2_pre_type=${v2_pre%%.*}
    local v1_pre_num=${v1_pre##*.}
    local v2_pre_num=${v2_pre##*.}

    if [ "$v1_pre_type" != "$v2_pre_type" ]; then
        case "$v1_pre_type-$v2_pre_type" in
            alpha-beta|alpha-rc|beta-rc) return 1 ;;
            beta-alpha|rc-alpha|rc-beta) return 0 ;;
            *) return 1 ;;
        esac
    fi

    if [ "$v1_pre_num" -ge "$v2_pre_num" ]; then return 0; else return 1; fi
}
