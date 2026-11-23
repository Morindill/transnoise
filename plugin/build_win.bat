@echo off
rem Build the VST3 on Windows (VS 2026 Community bundled CMake 4.x + Ninja). ASCII only: cmd mis-parses UTF-8 in .bat files.
rem Everything stays inside the project: sources in plugin\, build tree and JUCE checkout in plugin\build\ (git-ignored).
rem Run it from cmd; pushd maps the \\wsl.localhost UNC path to a temporary drive letter, which cmd/MSVC/Ninja need:
rem     cmd /c "pushd \\wsl.localhost\Ubuntu\home\morindill\noise_cancel\plugin && .\build_win.bat"
rem Output: plugin\build\Transnoise_artefacts\Release\VST3\Transnoise.vst3
setlocal
set "VS=C:\Program Files\Microsoft Visual Studio\18\Community"
set "CMAKE=%VS%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "HERE=%~dp0"

if not exist "%HERE%..\cpp\gen\weights.bin" (
  echo Missing cpp\gen\weights.bin: run deploy.export and deploy.pack in WSL first, see the header of cpp\Makefile
  exit /b 1
)
call "%VS%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
"%CMAKE%" -S "%HERE%." -B "%HERE%build" -G Ninja -DCMAKE_BUILD_TYPE=Release || exit /b 1
"%CMAKE%" --build "%HERE%build" || exit /b 1
echo.
echo Build finished. VST3:
dir /s /b "%HERE%build\*.vst3" 2>nul
endlocal
