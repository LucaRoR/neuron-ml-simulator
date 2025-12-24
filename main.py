from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from ml_app.gui.main_window import MainWindow

def _apply_styles(app: QApplication) -> None:
    """Loads styles and applies them to the app. Designed to work during testing."""
    repo_root = Path(__file__).resolve().parent
    qss_path = repo_root / "ml_app" / "resources" / "styles.qss"
    if qss_path.exists():
        app.setStyleSheet(qss_path.read_text(encoding="utf-8"))

def _set_app_icon(app: QApplication):
    """Sets app icon. Designed to work during testing."""
    repo_root = Path(__file__).resolve().parent
    icon_path = repo_root / "ml_app" / "resources" / "icons" / "app_icon.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

def main() -> int:
    app = QApplication(sys.argv)

    _apply_styles(app)
    _set_app_icon(app)

    win = MainWindow()
    win.show()

    return app.exec()

if __name__ == "__main__":
    raise SystemExit(main())