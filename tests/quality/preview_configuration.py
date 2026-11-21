"""An actual CMake cache and its effective preview state for tooling regressions."""

from pathlib import Path
import subprocess


PREVIEW_CASES = ("ON", "OFF", "TRUE", "1")


def configure_preview_option(root: Path, build: Path, value: str) -> bool:
    source = root / "preview-option"
    source.mkdir(exist_ok=True)
    (source / "CMakeLists.txt").write_text(
        """cmake_minimum_required(VERSION 4.3)
project(PreviewOption NONE)
option(JUICER_GRAIN_PREVIEW "Build the private Portra grain preview" OFF)
if(JUICER_GRAIN_PREVIEW)
    file(WRITE "${CMAKE_BINARY_DIR}/preview-enabled.txt" "yes")
else()
    file(WRITE "${CMAKE_BINARY_DIR}/preview-enabled.txt" "no")
endif()
""",
        encoding="utf-8",
    )
    command = ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja", "-DJUICER_GRAIN_PREVIEW:BOOL=" + value]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return (build / "preview-enabled.txt").read_text(encoding="utf-8") == "yes"
