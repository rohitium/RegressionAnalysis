"""
Convert Stanford HIVDB DRM files to RegressionAnalysis format.

This script reads DRM files with scores from raw/ and creates
DRM_{drug_class}.csv files with Pos and Mut columns.
"""

import re
import pandas as pd

# Drug class files to process
DRUG_CLASSES = ['NRTI', 'NNRTI', 'INSTI', 'PI']

def parse_mutation(rule):
    """Parse mutation string into position and amino acid.

    Examples:
        M41L -> (41, L)
        T69ins -> (69, #)
        K65del -> (65, ~)

    Args:
        rule: Mutation string (e.g., "M41L", "T69ins")

    Returns:
        Tuple of (position, mutation) or None if parsing fails
    """
    # Handle insertions
    if 'ins' in rule:
        match = re.match(r'[A-Z](\d+)ins', rule)
        if match:
            return (int(match.group(1)), '#')

    # Handle deletions
    if 'del' in rule:
        match = re.match(r'[A-Z](\d+)del', rule)
        if match:
            return (int(match.group(1)), '~')

    # Handle standard mutations (e.g., M41L)
    match = re.match(r'[A-Z](\d+)([A-Z])', rule)
    if match:
        return (int(match.group(1)), match.group(2))

    return None


def convert_drm_file(drug_class):
    """Convert a DRM file to the format needed for RegressionAnalysis.

    Args:
        drug_class: Drug class name (e.g., 'NRTI', 'NNRTI')
    """
    input_file = f'raw/{drug_class}.csv'
    output_file = f'DRM_{drug_class}.csv'

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Parse mutations
    mutations = []
    for rule in df['Rule']:
        parsed = parse_mutation(rule)
        if parsed:
            pos, mut = parsed
            mutations.append({'Pos': pos, 'Mut': mut})
        else:
            print(f"Warning: Could not parse mutation '{rule}' in {drug_class}")

    # Create output DataFrame
    result_df = pd.DataFrame(mutations)

    # Remove duplicates (same mutation may appear multiple times)
    result_df = result_df.drop_duplicates()

    # Sort by position
    result_df = result_df.sort_values('Pos')

    # Save to CSV
    result_df.to_csv(output_file, index=False)

    print(f"Created {output_file}: {len(result_df)} unique mutations")


def main():
    """Convert all DRM files."""
    for drug_class in DRUG_CLASSES:
        convert_drm_file(drug_class)

    print("\nAll DRM files converted successfully!")


if __name__ == "__main__":
    main()
