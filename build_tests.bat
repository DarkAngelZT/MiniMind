@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo ERROR: Please provide the llvm-mingw bin path.
    echo Usage: build_tests.bat LLVM_BIN_PATH [jobs]
    exit /b 2
)

set "LLVM_BIN=%~1"
set "JOBS=%NUMBER_OF_PROCESSORS%"
if not "%~2"=="" set "JOBS=%~2"
set "PATH=%LLVM_BIN%;%PATH%"

cmake -S . -B build -G "MinGW Makefiles" ^
    -DCMAKE_BUILD_TYPE=Debug ^
    -DCMAKE_C_COMPILER="%LLVM_BIN%\clang.exe" ^
    -DCMAKE_CXX_COMPILER="%LLVM_BIN%\clang++.exe" ^
    -DCMAKE_MAKE_PROGRAM="%LLVM_BIN%\mingw32-make.exe" ^
    -DMINIMIND_BUILD_TESTS=ON
if errorlevel 1 exit /b %ERRORLEVEL%

cmake --build build --target minimind_loss_test minimind_attention_test minimind_gru_test minimind_state_pooling_test minimind_embedding_test --parallel %JOBS%
if errorlevel 1 exit /b %ERRORLEVEL%

ctest --test-dir build --output-on-failure
exit /b %ERRORLEVEL%
