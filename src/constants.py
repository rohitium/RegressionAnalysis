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

# Stanford HIVDB dataset URLs by drug class (full datasets with expanded metadata)
DATA_URLS = {
    "PI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/PI_DataSet.Full.txt",
    "NRTI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/NRTI_DataSet.Full.txt",
    "NNRTI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/NNRTI_DataSet.Full.txt",
    "INSTI": "http://hivdb.stanford.edu/download/GenoPhenoDatasets/INI_DataSet.Full.txt",
}
