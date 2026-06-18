from __future__ import annotations

import os
import platform
from collections import namedtuple


def patch_windows_platform_for_torch() -> None:
    """Avoid slow/hanging WMI queries in platform.machine() during torch import on Windows."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if os.name != "nt":
        return

    arch = os.environ.get("PROCESSOR_ARCHITECTURE") or os.environ.get("PROCESSOR_ARCHITEW6432") or "AMD64"
    node = os.environ.get("COMPUTERNAME", "")
    uname_result = namedtuple("uname_result", ["system", "node", "release", "version", "machine", "processor"])

    platform.system = lambda: "Windows"  # type: ignore[assignment]
    platform.machine = lambda: arch  # type: ignore[assignment]
    platform.release = lambda: ""  # type: ignore[assignment]
    platform.version = lambda: ""  # type: ignore[assignment]
    platform.processor = lambda: arch  # type: ignore[assignment]
    platform.win32_ver = lambda release="", version="", csd="", ptype="": ("", "", "", "")  # type: ignore[assignment]
    platform.uname = lambda: uname_result("Windows", node, "", "", arch, arch)  # type: ignore[assignment]
