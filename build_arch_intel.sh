#!/usr/bin/env bash
set -euo pipefail

# Gamescope + FrameGen build script for Arch Linux on Intel Core i5-12450H (Alder Lake-H, Intel UHD)
# Installs into /usr/local/bin/gamescope so it does not conflict with the official pacman package.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build-alderlake"
PREFIX="${PREFIX:-/usr/local}"

echo "==> [1/4] Checking Arch Linux build dependencies..."
DEPS=(
  meson ninja pkgconf glslang vulkan-headers vulkan-icd-loader
  wayland wayland-protocols libdrm libinput libxkbcommon pixman
  sdl2 pipewire luajit libavif libcap libeis libdecor libdisplay-info
  libliftoff wlroots stb glm xorg-xwayland xorgproto
  libx11 libxcomposite libxdamage libxext libxfixes libxi libxmu
  libxrender libxres libxtst libxxf86vm libxcursor
  vulkan-intel mesa
)

if command -v pacman >/dev/null 2>&1; then
  MISSING=()
  for pkg in "${DEPS[@]}"; do
    if ! pacman -Qi "$pkg" >/dev/null 2>&1; then
      MISSING+=("$pkg")
    fi
  done
  if (( ${#MISSING[@]} > 0 )); then
    echo "Installing missing Arch packages: ${MISSING[*]}"
    sudo pacman -S --needed --noconfirm "${MISSING[@]}"
  else
    echo "All required Arch packages are already installed."
  fi
else
  echo "Warning: pacman not detected; make sure Meson, Ninja, glslang, Vulkan, Wayland, and wlroots dev headers are installed."
fi

echo "==> [2/4] Configuring Meson build (Intel Alder Lake-H optimizations)..."
export CFLAGS="-O3 -march=alderlake -mtune=alderlake"
export CXXFLAGS="-O3 -march=alderlake -mtune=alderlake"

if [[ -d "${BUILD_DIR}" ]]; then
  meson setup "${BUILD_DIR}" "${SCRIPT_DIR}" \
    --reconfigure \
    --prefix="${PREFIX}" \
    --buildtype=release \
    -Dpipewire=enabled \
    -Dbenchmark=disabled
else
  meson setup "${BUILD_DIR}" "${SCRIPT_DIR}" \
    --prefix="${PREFIX}" \
    --buildtype=release \
    -Dpipewire=enabled \
    -Dbenchmark=disabled
fi

echo "==> [3/4] Compiling gamescope + cs_framegen.comp SPIR-V shader..."
ninja -C "${BUILD_DIR}" -j"$(nproc)"

echo "==> [4/4] Installing to ${PREFIX}/bin/gamescope..."
sudo ninja -C "${BUILD_DIR}" install

if command -v setcap >/dev/null 2>&1; then
  sudo setcap 'cap_sys_nice=eip' "${PREFIX}/bin/gamescope" || true
fi

echo ""
echo "Build complete! Binary installed at: ${PREFIX}/bin/gamescope"
echo "Verify flags with: ${PREFIX}/bin/gamescope --help | grep -A 6 frame-gen"
