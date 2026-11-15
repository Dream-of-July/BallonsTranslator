#!/bin/sh
# Builds a release binary and assembles build/MangaTranslator.app (ad-hoc signed).
# Needs only the Swift toolchain; no network access.
set -eu

cd "$(dirname "$0")/.."
APP="build/MangaTranslator.app"

swift build -c release --product MangaTranslator
BIN_DIR="$(swift build -c release --show-bin-path)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN_DIR/MangaTranslator" "$APP/Contents/MacOS/MangaTranslator"
cp Resources/Info.plist "$APP/Contents/Info.plist"
cp -R Resources/en.lproj Resources/zh-Hans.lproj "$APP/Contents/Resources/"
plutil -lint "$APP/Contents/Info.plist" "$APP"/Contents/Resources/*.lproj/*.strings >/dev/null

codesign --force --sign - "$APP"
echo "Built $APP"
