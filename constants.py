"""
Configuration constants for HIVDB regression analysis.

Defines drug classes and data sources.
"""

# Drug classes and their associated drugs
DRUGS_BY_DRUG_CLASS = {
    "NRTI": ["3TC", "ABC", "AZT", "TFV"],
    "NNRTI": ["EFV", "RPV", "DOR"],
    "INSTI": ["DTG", "BIC", "CAB"],
    "PI": ["LPV", "ATV", "DRV"]
}

# Drug aliases - maps alternative names to canonical drug names
DRUG_ALIASES = {
    "TDF": "TFV",  # Tenofovir disoproxil fumarate
    "TAF": "TFV",  # Tenofovir alafenamide
}

# Mutation code keywords
INSERTION_KEYWORDS = {"#", "ins", "i"}
DELETION_KEYWORDS = {"~", "del", "d"}

# Stanford HIVDB dataset URLs by drug class
DATA_URLS = {
    "PI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.txt",
    "NRTI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/NRTI_DataSet.txt",
    "NNRTI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/NNRTI_DataSet.txt",
    "INSTI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/INI_DataSet.txt",
}

# Column index ranges for position data in each dataset
# Format: (start_index, end_index_exclusive) for pandas iloc
POSITION_SLICES = {
    "PI": (9, 108),      # Protease positions 1-99
    "NRTI": (7, 247),    # RT positions 1-240
    "NNRTI": (6, 324),   # RT positions 1-318
    "INSTI": (6, 294)    # Integrase positions 1-288
}
