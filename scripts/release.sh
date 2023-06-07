#!/bin/bash

print_help() {
    echo "Usage: $0 <version_number>"
    echo
    echo "This script requires a version number as an argument."
    echo "Example:"
    echo "$0 1.0.0"
}

if [ $# -eq 0 ]; then
    print_help
    exit 1
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    print_help
    exit 0
fi

VERSION=$1

# Strip off the leading "v" if it exists
if [[ $VERSION == v* ]]; then
    VERSION=${VERSION#v}
fi

BRANCH="release/v$VERSION"

git checkout -b $BRANCH \
&& yarn build \
&& yarn bump $VERSION \
&& git add . \
&& git commit -m "Release $VERSION" \
&& git tag v$VERSION \
&& git push origin $BRANCH -u \

yarn archive
