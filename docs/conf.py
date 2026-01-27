# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'equiflow'
copyright = '2024, João Matos, Jacob Ellen, Pedro Moreira'
author = 'João Matos, Jacob Ellen, Pedro Moreira'  # Fixed: should be 'author' (singular) as a string

release = '0.1.7'
version = '0.1.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

autosummary_generate = True

autoapi_dirs = ['../equiflow']
autoapi_generate_api_docs = True
autoapi_options = ['members', 'undoc-members', 'special-members', ]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'

html_theme_options = {
    "show_version_warning_banner": True,
    "navigation_depth": 5,
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 5,
    "show_prev_next": False,
}

html_context = {
    "github_user": "MoreiraP12",
    "github_repo": "equiflow-v2",
    "github_version": "main",
    "doc_path": "docs",
}

# -- Options for EPUB output -------------------------------------------------
epub_show_urls = 'footnote'
