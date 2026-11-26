# Microsoft Visual C++ runtime

Windows builds use the Microsoft DLL runtime with the selected Visual Studio
18 x64 toolset. CMake's `InstallRequiredSystemLibraries` takes the x64
redistributable DLLs from that Visual Studio installation. The Windows Universal
CRT is supplied by the supported Windows system.

The redistribution terms are provided by the Visual Studio license and its
[distributable code list](https://learn.microsoft.com/en-us/visualstudio/releases/2026/redistribution).
