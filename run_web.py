"""
ssook Desktop Launcher

Starts the FastAPI server and opens the UI.
Supports two modes:
  - pywebview (native window, if installed)
  - browser fallback (opens default browser)

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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# PyInstaller frozen exe may set sys.stdout/stderr to None → patch early
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")


def _set_windows_icon():
    """Set taskbar/titlebar icon on Windows."""
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        # Set AppUserModelID so Windows groups this as its own app with custom icon
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('ssook.App')
        # Set console window icon
        ico = str(ROOT / "assets" / "icon.ico")
        if os.path.exists(ico):
            from ctypes import wintypes
            SendMessage = ctypes.windll.user32.SendMessageW
            LoadImage = ctypes.windll.user32.LoadImageW
            IMAGE_ICON, LR_LOADFROMFILE = 1, 0x0010
            WM_SETICON, ICON_BIG, ICON_SMALL = 0x0080, 1, 0
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                for size_flag, px in [(ICON_BIG, 32), (ICON_SMALL, 16)]:
                    hicon = LoadImage(0, ico, IMAGE_ICON, px, px, LR_LOADFROMFILE)
                    if hicon:
                        SendMessage(hwnd, WM_SETICON, size_flag, hicon)
    except Exception:
        pass


def start_server(port: int):
    """Start FastAPI server in a background thread."""
    import uvicorn
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        filename="ssook.log", filemode="a")
    # Also log to stderr
    logging.getLogger().addHandler(logging.StreamHandler())
    try:
        from server import app
        uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning",
                    timeout_keep_alive=30)
    except Exception as e:
        logging.exception(f"Server crashed: {e}")
        print(f"\n[FATAL] Server error: {e}", file=sys.stderr)


def wait_for_server(port: int, timeout: float = 10.0):
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
    server_thread = threading.Thread(target=start_server, args=(args.port,), daemon=True)
    server_thread.start()

    print(f"Starting ssook at {url} ...")
    if not wait_for_server(args.port):
        print("ERROR: Server failed to start")
        sys.exit(1)

    # Try pywebview for native desktop window
    if not args.browser:
        try:
            import webview
            print("Opening native window...")
            ico = str(ROOT / "assets" / ("icon.ico" if sys.platform == "win32" else "icon.png"))
            webview.create_window(
                "ssook",
                url,
                width=1400,
                height=900,
                min_size=(1024, 600),
            )
            webview.start(icon=ico)
            return
        except ImportError:
            pass
        except Exception as e:
            print(f"[pywebview error] {e}", file=sys.stderr)
            import traceback; traceback.print_exc()

    # Fallback: open in browser
    print(f"Opening browser at {url}")
    webbrowser.open(url)

    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
