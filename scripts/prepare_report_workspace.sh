#!/usr/bin/env bash
# Restore the original flat report layout in a separate copy for historical experiments.
set -euo pipefail

if [[ $# != 1 || -z ${1:-} ]]; then
    echo "Usage: $0 NEW_DIRECTORY" >&2
    exit 2
fi

script_path=$(realpath -- "${BASH_SOURCE[0]}")
root=$(dirname -- "$(dirname -- "$script_path")")
archive="$root/artifacts/history/report-layout.tar.gz"
if [[ ! -f $archive ]]; then
    echo "Historical report archive is missing: $archive" >&2
    exit 1
fi

if [[ -e $1 || -L $1 ]]; then
    echo "Choose a new directory. The requested path already exists: $1" >&2
    exit 1
fi
destination=$(realpath -m -- "$1")
if [[ $destination == "$root" || $destination == "$root"/* ]]; then
    echo "Choose a directory outside the project: $destination" >&2
    exit 1
fi

mkdir -- "$destination"
for item in "$root"/*; do
    case "${item##*/}" in
        dist) continue ;;
    esac
    cp -a --reflink=auto -- "$item" "$destination/"
done
cp --reflink=auto -- "$root/README_en.md" "$destination/README.md"
tar -xzf "$archive" -C "$destination"

# Staged runtimes use absolute links to runs and training data. Keep their writes in this copy.
while IFS= read -r -d '' link; do
    previous=$(readlink -- "$link")
    if [[ $previous == "$root"/* ]]; then
        ln -sfnT -- "$destination/${previous#"$root"/}" "$link"
    fi
done < <(find "$destination" -type l -print0)
if [[ -d $root/.venv ]]; then
    ln -s -- "$root/.venv" "$destination/.venv"
fi
printf 'Historical report workspace ready at %s\n' "$destination"
