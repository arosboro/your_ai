# Copyright © 2023 Apple Inc.

# -*- coding: utf-8 -*-

"""Sphinx configuration file for MLX C API documentation.

This module configures the Sphinx documentation builder for the MLX C library.
It sets up project metadata, extensions, themes, and build options.

The configuration requires mlx.core to be installed in the environment, as it
is imported to access version and metadata information during the doc build.

Usage:
    This file is automatically invoked by Sphinx during documentation builds:
        sphinx-build -b html docs/src docs/build
"""

import os
import subprocess

import mlx.core as mx

# -- Project information -----------------------------------------------------

project = "MLX C"
copyright = "2023-2025, MLX Contributors"
author = "MLX Contributors"
version = "0.2.0"
release = version

# -- General configuration ---------------------------------------------------

extensions = ["breathe"]
breathe_projects = {"mlxc" : "../build/xml"}
breathe_default_project = "mlxc"
templates_path = ["_templates"]
html_static_path = ["_static"]
source_suffix = ".rst"
master_doc = "index"
highlight_language = "c"
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"

html_theme_options = {
    "show_toc_level": 2,
    "repository_url": "https://github.com/ml-explore/mlx-c",
    "use_repository_button": True,
    "navigation_with_keys": False,
    "logo": {
        "image_light": "_static/mlx_logo.png",
        "image_dark": "_static/mlx_logo_dark.png",
    },
}


# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "mlxc_doc"
