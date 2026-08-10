# Put the SBGrid collection on $PATH when it is mounted.
#
# SBGrid is a site installation on shared storage, which actl mounts read-only
# at /programs for every diffuse workspace. It is not part of this image, so
# this script must be a no-op whenever the mount is absent (any non-diffuse
# workspace). Every step is guarded accordingly: without the mount, /programs
# is an ordinary empty directory.
#
# Set SBGRID_NO_AUTOINIT=1 before shell start to opt out.

if [ -z "${SBGRID_NO_AUTOINIT:-}" ] && [ -z "${SBGRID_INITIALISED:-}" ] && [ -r /programs/sbgrid.shrc ]; then
    SBGRID_INITIALISED=1
    export SBGRID_INITIALISED
    # sbgrid.shrc runs commands that return non-zero benignly, so it aborts
    # under `set -e` and would take the whole shell with it. It is also chatty
    # on first run and writes into $HOME. Neither should be able to break shell
    # startup, hence the redirect and the `|| true`.
    . /programs/sbgrid.shrc >/dev/null 2>&1 || true
fi
