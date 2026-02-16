"""Embodied Claude Installer GUI entrypoint."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QWizard


def _build_installer_wizard_class() -> type["QWizard"]:
    """Build installer wizard class with lazy PyQt/page imports."""
    from PyQt6.QtWidgets import QWizard

    from .pages.api_key import ApiKeyPage
    from .pages.camera import CameraSelectionPage
    from .pages.complete import CompletePage
    from .pages.dependencies import DependenciesPage
    from .pages.install import InstallationPage
    from .pages.welcome import WelcomePage

    class EmbodiedClaudeInstaller(QWizard):
        """Main installer wizard."""

        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Embodied Claude Installer")
            self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
            self.setOption(QWizard.WizardOption.HaveHelpButton, False)
            self.setMinimumSize(800, 600)

            self.addPage(WelcomePage())
            self.addPage(DependenciesPage())
            self.addPage(CameraSelectionPage())
            self.addPage(ApiKeyPage())
            self.addPage(InstallationPage())
            self.addPage(CompletePage())

    return EmbodiedClaudeInstaller


def main() -> None:
    """Entry point for the installer."""
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise RuntimeError(
            "PyQt6 runtime dependencies are missing. Install GUI dependencies "
            "before launching installer."
        ) from exc

    installer_class = _build_installer_wizard_class()
    app = QApplication(sys.argv)
    app.setApplicationName("Embodied Claude Installer")

    wizard = installer_class()
    wizard.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
