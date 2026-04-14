"""
ssook Desktop Launcher

Starts the FastAPI server and opens the UI in a pywebview window.
Falls back to the default browser if pywebview is not installed.

Usage:
  python run_web.py              # auto-detect best mode
  python run_web.py --browser    # force browser mode
  python run_web.py --port 9000  # custom port
"""
import os
import sys
import time
import argparse
import threading
import webbrowser
from pathlib import Path

# ── Frozen exe support ──────────────────────────────────
# PyInstaller with console=False sets stdout/stderr to None
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

if getattr(sys, "frozen", False):
    ROOT = Path(sys._MEIPASS)
    os.chdir(os.path.dirname(sys.executable))
else:
    ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT))

# ── EP 런타임 선택 (onnxruntime import 전에 실행) ──
from core.ep_selector import select_and_activate
_selected_ep = select_and_activate()


def _set_windows_icon():
    """Set taskbar/titlebar icon on Windows."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("ssook.App")
        ico = str(ROOT / "assets" / "icon.ico")
        if os.path.exists(ico):
            SendMessage = ctypes.windll.user32.SendMessageW
            LoadImage = ctypes.windll.user32.LoadImageW
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                for flag, px in [(1, 32), (0, 16)]:
                    h = LoadImage(0, ico, 1, px, px, 0x0010)
                    if h:
                        SendMessage(hwnd, 0x0080, flag, h)
    except Exception:
        pass


def start_server(port: int):
    """Start FastAPI server in a background thread."""
    import logging
    log_path = os.path.join(os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else ".", "ssook.log")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        filename=log_path, filemode="a")
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        import uvicorn
        from server import app
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning",
                    timeout_keep_alive=30)
    except Exception as e:
        logging.exception(f"Server crashed: {e}")


def wait_for_server(port: int, timeout: float = 15.0):
    """Wait until the server is accepting connections."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main():
    parser = argparse.ArgumentParser(description="ssook Desktop Launcher")
    parser.add_argument("--browser", action="store_true", help="Force browser mode")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}"
    _set_windows_icon()

    # Start server in background
    threading.Thread(target=start_server, args=(args.port,), daemon=True).start()

    if not wait_for_server(args.port):
        print("ERROR: Server failed to start. Check ssook.log", file=sys.stderr)
        sys.exit(1)

    # pywebview (native window) — default
    if not args.browser:
        try:
            import webview
            import logging
            logging.getLogger("pywebview").setLevel(logging.DEBUG)
            ico = str(ROOT / "assets" / ("icon.ico" if sys.platform == "win32" else "icon.png"))
            window = webview.create_window("ssook", url, width=1400, height=900, min_size=(1024, 600),
                                           text_select=True)

            def _on_loaded():
                """Fix keyboard focus in PyInstaller frozen exe (console=False)."""
                try:
                    window.evaluate_js('document.body.focus();')
                    if sys.platform == 'win32':
                        import ctypes
                        hwnd = ctypes.windll.user32.GetForegroundWindow()
                        if hwnd:
                            ctypes.windll.user32.SetFocus(hwnd)
                except Exception:
                    pass

            window.events.loaded += _on_loaded
            webview.start(icon=ico, debug=False)
            return
        except ImportError as e:
            print(f"[pywebview] not installed: {e}", file=sys.stderr)
        except Exception as e:
            print(f"[pywebview] failed: {e}", file=sys.stderr)
            import traceback; traceback.print_exc(file=sys.stderr)

    # Fallback: browser
    webbrowser.open(url)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
