"""Data loading and processing utilities for the dashboard.

This module provides functions for loading, filtering, and validating the
protein interaction data used in the Streamlit application.
"""
import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    """Load protein interaction datasets with error handling.

    This function loads the direct and indirect protein interaction data from
    CSV files located in the `data/` directory. It includes error handling to
    manage missing files by creating empty DataFrames with the expected
    column structure. The results are cached using Streamlit's caching
    mechanism to improve performance.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two pandas
            DataFrames. The first DataFrame holds the direct interaction
            data, and the second contains the indirect effects data.
    """
    try:
        # Load direct interactions
        direct_path = "data/EGCG_Direct_Interactions.csv"
        if os.path.exists(direct_path):
            direct_df = pd.read_csv(direct_path)
        else:
            # Create empty DataFrame with expected columns if file doesn't exist
            direct_df = pd.DataFrame(columns=[
                'protein', 'gene', 'interaction_type', 'affinity', 
                'evidence_category', 'species', 'disease_context'
            ])
        
        # Load indirect effects
        indirect_path = "data/EGCG_Indirect_Effects.csv"
        if os.path.exists(indirect_path):
            indirect_df = pd.read_csv(indirect_path)
        else:
            # Create empty DataFrame with expected columns if file doesn't exist
            indirect_df = pd.DataFrame(columns=[
                'protein', 'gene', 'effect_type', 'magnitude', 
                'evidence_category', 'species', 'disease_context'
            ])
        
        return direct_df, indirect_df
    
    except Exception as e:
        st.error(f"Error loading data files: {str(e)}")
        # Return empty DataFrames as fallback
        empty_direct = pd.DataFrame(columns=[
            'protein', 'gene', 'interaction_type', 'affinity', 
            'evidence_category', 'species', 'disease_context'
        ])
        empty_indirect = pd.DataFrame(columns=[
            'protein', 'gene', 'effect_type', 'magnitude', 
            'evidence_category', 'species', 'disease_context'
        ])
        return empty_direct, empty_indirect

def get_known_interactions(df, query_protein, species=None, disease_contexts=None):
    """Filter interactions based on query parameters.

    This function filters a DataFrame of protein interactions based on a query
    protein, species, and a list of disease contexts. The filtering is
    case-insensitive.

    Args:
        df (pd.DataFrame): The DataFrame to filter.
        query_protein (str): The protein or gene to search for.
        species (str, optional): The species to filter by. Defaults to None.
        disease_contexts (list[str], optional): A list of disease contexts to
            filter by. Defaults to None.

    Returns:
        pd.DataFrame: The filtered DataFrame.
    """
    if df.empty:
        return df
    
    # Filter by query protein (case-insensitive)
    if query_protein:
        mask = (
            df['protein'].str.contains(query_protein, case=False, na=False) |
            df['gene'].str.contains(query_protein, case=False, na=False)
        )
        filtered_df = df[mask]
    else:
        filtered_df = df.copy()
    
    # Filter by species
    if species and 'species' in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df['species'].str.contains(species, case=False, na=False)
        ]
    
    # Filter by disease contexts
    if disease_contexts is not None and 'disease_context' in filtered_df.columns:
        if not disease_contexts:  # Empty list means no contexts selected - show no data
            filtered_df = filtered_df.iloc[0:0]  # Return empty DataFrame with same structure
        else:
            disease_mask = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            for disease in disease_contexts:
                disease_mask |= filtered_df['disease_context'].str.contains(
                    disease, case=False, na=False
                )
            filtered_df = filtered_df[disease_mask]
    
    return filtered_df.copy()

def validate_data_format(df, expected_columns):
    """Validate that DataFrame has expected columns.

    This function checks if a DataFrame contains a set of expected columns.
    If any columns are missing, it adds them to the DataFrame with `pd.NA`
    values and displays a warning in the Streamlit app.

    Args:
        df (pd.DataFrame): The DataFrame to validate.
        expected_columns (list[str]): A list of column names that are
            expected to be in the DataFrame.

    Returns:
        pd.DataFrame: The validated (and possibly modified) DataFrame.
    """
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        st.warning(f"Missing columns in data: {missing_columns}")
        # Add missing columns with NaN values
        for col in missing_columns:
            df[col] = pd.NA
    
    return df

def get_summary_stats(direct_df, indirect_df, query_protein):
    """Calculate summary statistics for the dashboard.

    This function computes a set of summary statistics from the direct and
    indirect interaction DataFrames. These statistics are used to populate
    the metric displays on the dashboard.

    Args:
        direct_df (pd.DataFrame): DataFrame of direct interactions.
        indirect_df (pd.DataFrame): DataFrame of indirect effects.
        query_protein (str): The protein or gene being queried.

    Returns:
        dict[str, any]: A dictionary containing summary statistics,
            including counts of direct and indirect interactions, the number
            of unique proteins, the average affinity, and the top evidence
            category.
    """
    stats = {
        'direct_count': len(direct_df),
        'indirect_count': len(indirect_df),
        'unique_proteins': 0,
        'avg_affinity': 0,
        'top_evidence_category': 'N/A'
    }
    
    if not direct_df.empty:
        stats['unique_proteins'] = direct_df['protein'].nunique()
        if 'affinity' in direct_df.columns:
            stats['avg_affinity'] = direct_df['affinity'].mean()
        if 'evidence_category' in direct_df.columns:
            stats['top_evidence_category'] = direct_df['evidence_category'].value_counts().index[0]
    
    return stats
