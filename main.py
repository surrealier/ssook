"""진입점"""
import os
import sys

# 실행 파일 기준 경로 설정 (PyInstaller 번들 대응)
if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS  # type: ignore
    os.chdir(os.path.dirname(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(BASE_DIR)

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setAttribute(Qt.AA_EnableHighDpiScaling if hasattr(Qt, "AA_EnableHighDpiScaling") else Qt.ApplicationAttribute(14))
    app.setStyle("Fusion")

    icon_path = os.path.join(BASE_DIR, "assets", "icon.ico")
    if os.path.isfile(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
