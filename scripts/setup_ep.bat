@echo off
setlocal
echo ============================================================
echo  EP Package Setup — onnxruntime 변종별 격리 설치
echo ============================================================
echo.
echo 각 EP 패키지를 ep_packages\{key} 에 독립적으로 설치합니다.
echo 기존 Python 환경의 onnxruntime에는 영향을 주지 않습니다.
echo.

set BASE=%~dp0..\ep_packages

echo [1/4] CPU — onnxruntime
mkdir "%BASE%\cpu" 2>nul
pip install onnxruntime --target "%BASE%\cpu" --upgrade
echo.

echo [2/4] CUDA / TensorRT — onnxruntime-gpu
mkdir "%BASE%\cuda" 2>nul
pip install onnxruntime-gpu --target "%BASE%\cuda" --upgrade
echo.

echo [3/4] OpenVINO — onnxruntime-openvino
mkdir "%BASE%\openvino" 2>nul
pip install onnxruntime-openvino --target "%BASE%\openvino" --upgrade
echo.

echo [4/4] DirectML — onnxruntime-directml
mkdir "%BASE%\directml" 2>nul
pip install onnxruntime-directml --target "%BASE%\directml" --upgrade
echo.

echo ============================================================
echo  설치 완료. 앱을 재시작하면 EP 선택이 활성화됩니다.
echo ============================================================
pause
