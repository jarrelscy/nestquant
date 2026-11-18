#!/bin/zsh
# Build "NQ Widget.app" (the NestQuant server desktop widget) and install it in ~/Applications.
# Needs only the Xcode Command Line Tools (swiftc). Usage: macos/widget/build.sh [install-dir]
set -e
HERE=${0:A:h}
DEST=${1:-$HOME/Applications}
APP="$DEST/NQ Widget.app"
mkdir -p "$APP/Contents/MacOS"
swiftc -O -swift-version 5 -o "$APP/Contents/MacOS/NQWidget" "$HERE/NQWidget.swift" -framework AppKit -framework IOKit
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>ai.nestquant.widget</string>
  <key>CFBundleName</key><string>NQ Widget</string>
  <key>CFBundleExecutable</key><string>NQWidget</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSUIElement</key><true/>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSAppTransportSecurity</key><dict><key>NSAllowsLocalNetworking</key><true/><key>NSAllowsArbitraryLoads</key><true/></dict>
</dict></plist>
PLIST
codesign --force --sign - "$APP" >/dev/null 2>&1 || true
echo "Built $APP"
echo "Start it with: open \"$APP\"   (right-click the widget for options; add it to Login Items to start at login)"
