"""Compatibility layer for the frozen ``joblib`` preprocessors.

The paper artifacts were serialized when :class:`FoldPreprocessor` lived in a
module named ``grouped_common``.  Re-exporting the public preprocessing API
under that historical module name keeps the frozen artifacts portable without
duplicating scientific logic.
"""

from preprocessing import *  # noqa: F401,F403

