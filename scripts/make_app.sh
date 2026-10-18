#!/bin/sh
# Build a release binary and wrap it in a double-clickable macOS app bundle:
#   scripts/make_app.sh            -> dist/Polymix.app
set -e
cd "$(dirname "$0")/.."
cargo build -p polymix --release
APP=dist/Polymix.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/polymix "$APP/Contents/MacOS/polymix"
# App icon from assets/icon_1024.png (regenerate with: cargo run -p polymix --example gen_icon -- assets/icon_1024.png)
if [ -f assets/icon_1024.png ] && command -v iconutil >/dev/null; then
  ICONSET=$(mktemp -d)/Polymix.iconset
  mkdir -p "$ICONSET"
  for s in 16 32 128 256 512; do
    sips -z $s $s assets/icon_1024.png --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
    d=$((s * 2))
    sips -z $d $d assets/icon_1024.png --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/Polymix.icns"
fi
VERSION=$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Polymix</string>
  <key>CFBundleDisplayName</key><string>Polymix</string>
  <key>CFBundleIdentifier</key><string>org.polymix.Polymix</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleExecutable</key><string>polymix</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleIconFile</key><string>Polymix</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>Mesh</string>
      <key>CFBundleTypeRole</key><string>Editor</string>
      <key>CFBundleTypeExtensions</key>
      <array><string>stl</string><string>obj</string><string>ply</string><string>off</string><string>3mf</string></array>
    </dict>
    <dict>
      <key>CFBundleTypeName</key><string>Medical image</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <!-- Listed under Open With without taking over these files from other viewers. -->
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>CFBundleTypeExtensions</key>
      <array><string>nii</string><string>gz</string><string>dcm</string><string>ima</string><string>nrrd</string><string>nhdr</string><string>hdr</string><string>mha</string><string>mhd</string><string>mgh</string><string>mgz</string><string>tif</string><string>tiff</string></array>
    </dict>
    <dict>
      <key>CFBundleTypeName</key><string>Folder of images</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>LSItemContentTypes</key>
      <array><string>public.folder</string></array>
    </dict>
  </array>
</dict>
</plist>
PLIST
# Ad-hoc signature so Gatekeeper allows running the locally built app.
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
echo "Built $APP"
