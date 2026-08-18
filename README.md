# DRIFT-SENSE V24

## Overview

DRIFT-SENSE V24 is a synthetic image localization system designed to identify the center coordinates of a reference pattern within a search image.

The final validated configuration uses spatial consistency and image-derived signals.

### Final validated performance

- Mean localization error: 7.88 px
- Median localization error: 7.36 px
- Within 5 px: 43.33%
- Within 10 px: 70.00%
- Within 20 px: 96.67%
- Test pairs: 30

The final model is the Step-8 Spatial Consistency configuration.

---

## Repository Structure

```text
drift-sense-v24/
├── README.md
├── requirements.txt
├── references.md
├── src/
│   ├── dataset_generator.py
│   └── localization_inference.py
└── generated_data/
