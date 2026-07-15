"""ReviewLens — Aspect-Based Review Intelligence.

A pipeline that extracts aspects from product/app reviews, classifies sentiment
*per aspect*, clusters aspects into themes, and surfaces the result in a
business dashboard.

Public entry points:
    reviewlens.pipeline.run_pipeline  -- end-to-end baseline pipeline
    reviewlens.config.load_config     -- load config.yaml
"""

__version__ = "0.1.0"
__author__ = "Saud Satopay"
