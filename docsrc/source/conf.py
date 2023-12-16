# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

project = 'RecWizard'
copyright = '2023, McAuley Lab'
author = 'McAuley Lab'
release = 'v0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["m2r",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_design"
]

source_suffix = ['.rst', '.md']
exclude_patterns = []
html_theme = "furo"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

autodoc_member_order = 'bysource'

html_title = "RecWizard"

html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "logo.png",
    "dark_logo": "logo.png",
}
