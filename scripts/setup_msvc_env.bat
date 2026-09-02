@echo off
REM ---------------------------------------------------------------------------
REM Phase L: MSVC environment bootstrap for torch.compile (inductor CPU backend).
REM
REM Why this exists: this machine has VS 2022 BuildTools AND the Windows 10 SDK
REM installed, but `vswhere.exe` is missing from the install. vcvarsall.bat uses
REM vswhere to locate the Windows SDK, so without it vcvars64.bat reports
REM "Environment initialized" while silently omitting the ucrt/um/shared INCLUDE
REM and LIB paths. The result is that cl.exe runs but dies with
REM   fatal error C1083: Cannot open include file: 'crtdbg.h'
REM which is exactly what torch.compile's inductor backend hit.
REM
REM This script calls vcvars64.bat for the compiler/PATH setup, then appends the
REM SDK include/lib directories explicitly so cl.exe can actually build the
REM generated C++ kernels.
REM ---------------------------------------------------------------------------

set "VS_BUILDTOOLS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
set "MSVC_VER=14.44.35207"
set "WIN_SDK=C:\Program Files (x86)\Windows Kits\10"
set "SDK_VER=10.0.22621.0"

call "%VS_BUILDTOOLS%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1

set "MSVC_ROOT=%VS_BUILDTOOLS%\VC\Tools\MSVC\%MSVC_VER%"

set "INCLUDE=%MSVC_ROOT%\include;%WIN_SDK%\Include\%SDK_VER%\ucrt;%WIN_SDK%\Include\%SDK_VER%\um;%WIN_SDK%\Include\%SDK_VER%\shared;%INCLUDE%"
set "LIB=%MSVC_ROOT%\lib\x64;%WIN_SDK%\Lib\%SDK_VER%\ucrt\x64;%WIN_SDK%\Lib\%SDK_VER%\um\x64;%LIB%"
set "PATH=%MSVC_ROOT%\bin\Hostx64\x64;%PATH%"
