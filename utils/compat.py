# datasets library compatibility fix: 'List' feature type was renamed to 'Sequence' in datasets >= 3.x
# Import this module before any code that calls datasets.load_from_disk()
from datasets.features.features import _FEATURE_TYPES
if "List" not in _FEATURE_TYPES:
    _FEATURE_TYPES["List"] = _FEATURE_TYPES["Sequence"]
