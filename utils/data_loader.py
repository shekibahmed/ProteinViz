import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_data():
    """Load protein interaction datasets with error handling"""
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
    """Filter interactions based on query parameters"""
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
    if disease_contexts and 'disease_context' in filtered_df.columns:
        disease_mask = pd.Series([False] * len(filtered_df))
        for disease in disease_contexts:
            disease_mask |= filtered_df['disease_context'].str.contains(
                disease, case=False, na=False
            )
        filtered_df = filtered_df[disease_mask]
    
    return filtered_df.copy()

def validate_data_format(df, expected_columns):
    """Validate that DataFrame has expected columns"""
    missing_columns = set(expected_columns) - set(df.columns)
    if missing_columns:
        st.warning(f"Missing columns in data: {missing_columns}")
        # Add missing columns with NaN values
        for col in missing_columns:
            df[col] = pd.NA
    
    return df

def get_summary_stats(direct_df, indirect_df, query_protein):
    """Calculate summary statistics for the dashboard"""
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
