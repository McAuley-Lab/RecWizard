"""A nox configuration file so that we can build the documentation easily with nox.
- see the README.md for information about nox.
- ref: https://nox.thea.codes/en/stable/
"""
import nox
from pathlib import Path

nox.options.reuse_existing_virtualenvs = True

build_command = ["-b", "html", "docs", "docs/_build/html"]


def _should_install(session: nox.Session) -> bool:
    """Decide if we should install an environment or if it already exists.

    This speeds up the local install considerably because building the wheel
    for this package takes some time.

    We assume that if `sphinx-build` is in the bin/ path, the environment is
    installed.

    Parameter:
        session: the current nox session
    """
    if session.bin_paths is None:
        session.log("Running with `--no-venv` so don't install anything...")
        return False
    bin_files = list(Path(session.bin).glob("*"))
    sphinx_is_installed = any("sphinx-build" in ii.name for ii in bin_files)
    force_reinstall = "reinstall" in session.posargs or "-r" in session.posargs
    should_install = not sphinx_is_installed or force_reinstall
    if should_install:
        session.log("Installing fresh environment...")
    else:
        session.log("Skipping environment install...")
    return should_install


@nox.session()
def docs(session):
    if _should_install(session):
        session.install("-e", ".[doc]")
        session.install("sphinx-theme-builder[cli]")
    if "live" in session.posargs:
        AUTOBUILD_IGNORE = [
            "*/.github/*",
            "*/_data/*",
            "*/howto/languages.rst",
            "*/howto/user_interface.rst",
            "*/howto/lab_workspaces.rst",
            "*/using/config_files.rst",
        ]
        cmd = ["sphinx-autobuild"]
        for folder in AUTOBUILD_IGNORE:
            cmd.extend(["--ignore", f"*/{folder}/*"])
        cmd.extend(build_command)
        session.run(*cmd)
    else:
        session.run("sphinx-build", *build_command)
