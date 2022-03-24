# Sphinx API Autodocumentation

## How to Build

Follow these steps if you want to build the API documentation.

### Prerequisites

You must have the following installed:
- sphinx
- sphinx_rtd_theme
- myst-parser

These are available via conda/conda-forge or pypi.

### If you added a module...

If you added an entire new module, then you must:
1. Create an .rst file for it in the docs/source/ folder
2. Add a link to it in the docs/source/index.rst file

Follow the formatting of the other module .rst files in docs/source/.
See the [Sphinx website](https://www.sphinx-doc.org/en/master/) for more information.

### Command

In this directory, simply type:
```bash
make html
```

## How to View

API documentation will be organized in a series of html files in docs/build/html/.
Open docs/build/html/index.html in your favorite browser to view.

This will re-build any html files that have un-built changes.  They will replace the files in docs/build/html/.
