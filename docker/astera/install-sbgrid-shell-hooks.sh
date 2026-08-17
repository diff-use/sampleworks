#!/usr/bin/env bash
set -euo pipefail

# Wire the guarded SBGrid init into every shell actl might open, mirroring
# install-ext-shell-hooks.sh. Idempotent: re-running never double-appends.

profile_script="/etc/profile.d/sbgrid-shell.sh"
profile_comment="# Sampleworks: put SBGrid on \$PATH when /programs is mounted."
profile_line="[ -r ${profile_script} ] && . ${profile_script}"

touch /root/.bashrc /home/dev/.bashrc

for profile_file in /etc/bash.bashrc /root/.bashrc /home/dev/.bashrc /etc/zsh/zshrc /etc/zsh/zprofile; do
    if [ ! -e "${profile_file}" ]; then
        continue
    fi
    if grep -Fqs "${profile_line}" "${profile_file}"; then
        continue
    fi
    printf '\n%s\n%s\n' "${profile_comment}" "${profile_line}" >> "${profile_file}"
done
