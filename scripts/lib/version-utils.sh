#!/bin/bash

# compare_versions v1 v2
# Returns 0 if v1 >= v2, 1 if v1 < v2.
# Accepts semver strings with optional -alpha.N / -beta.N / -rc.N prerelease.
compare_versions() {
    local v1=$1
    local v2=$2

    local v1_base=$(echo "$v1" | cut -d'-' -f1)
    local v2_base=$(echo "$v2" | cut -d'-' -f1)
    local v1_pre=$(echo "$v1" | cut -d'-' -f2- | sed "s/^$v1_base\$//")
    local v2_pre=$(echo "$v2" | cut -d'-' -f2- | sed "s/^$v2_base\$//")

    IFS='.' read -r v1_major v1_minor v1_patch <<< "$v1_base"
    IFS='.' read -r v2_major v2_minor v2_patch <<< "$v2_base"

    if [ "$v1_major" -gt "$v2_major" ]; then return 0; fi
    if [ "$v1_major" -lt "$v2_major" ]; then return 1; fi
    if [ "$v1_minor" -gt "$v2_minor" ]; then return 0; fi
    if [ "$v1_minor" -lt "$v2_minor" ]; then return 1; fi
    if [ "$v1_patch" -gt "$v2_patch" ]; then return 0; fi
    if [ "$v1_patch" -lt "$v2_patch" ]; then return 1; fi

    # No prerelease (stable) > prerelease
    if [ -z "$v1_pre" ] && [ -n "$v2_pre" ]; then return 0; fi
    if [ -n "$v1_pre" ] && [ -z "$v2_pre" ]; then return 1; fi

    # Both stable or both prerelease
    if [ -z "$v1_pre" ] && [ -z "$v2_pre" ]; then return 0; fi

    # alpha < beta < rc
    local v1_pre_type=$(echo "$v1_pre" | cut -d'.' -f1)
    local v2_pre_type=$(echo "$v2_pre" | cut -d'.' -f1)
    local v1_pre_num=$(echo "$v1_pre" | cut -d'.' -f2)
    local v2_pre_num=$(echo "$v2_pre" | cut -d'.' -f2)

    if [ "$v1_pre_type" != "$v2_pre_type" ]; then
        case "$v1_pre_type-$v2_pre_type" in
            alpha-beta|alpha-rc|beta-rc) return 1 ;;
            beta-alpha|rc-alpha|rc-beta) return 0 ;;
        esac
    fi

    if [ "$v1_pre_num" -ge "$v2_pre_num" ]; then return 0; else return 1; fi
}
