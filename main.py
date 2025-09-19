import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import base64
from utils.data_loader import load_data, get_known_interactions
from utils.ml_models import predict_interaction, get_available_models
from utils.protein_utils import get_protein_sequence, validate_protein_id, get_protein_info, get_alphafold_structure_url

# Configure page
st.set_page_config(
    page_title="Protein Interaction Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load and cache data
@st.cache_data
def initialize_data():
    """Load and cache the protein interaction datasets"""
    direct_df, indirect_df = load_data()
    return direct_df, indirect_df

# Main title
st.title("🧬 EGCG Protein Interaction Research Dashboard")
st.markdown("### Explore EGCG protein interactions and effects with ML-powered predictions")

# Sidebar controls
st.sidebar.header("Query Parameters")

# Protein query input
query_protein = st.sidebar.text_input(
    "Filter by Protein/Gene",
    value="",
    help="Enter a protein identifier to filter results (leave empty to show all EGCG interactions)"
)

# Species selection
species_options = ["Human", "Rat", "Mouse", "All Species"]
selected_species = st.sidebar.selectbox(
    "Species",
    species_options,
    index=0
)

# Disease context filters
st.sidebar.subheader("Disease Contexts")
disease_contexts = {
    "Neurodegenerative": st.sidebar.checkbox("Neurodegenerative", value=True),
    "Anti-aging": st.sidebar.checkbox("Anti-aging", value=True),
    "Gut": st.sidebar.checkbox("Gut", value=False),
    "Cardiovascular": st.sidebar.checkbox("Cardiovascular", value=True)
}

# Model selection for predictions
st.sidebar.subheader("Prediction Model")
available_models = get_available_models()
selected_model = st.sidebar.radio(
    "Choose Model",
    available_models,
    help="Select the machine learning model for interaction prediction"
)

# Advanced options in expander
with st.sidebar.expander("Advanced Options"):
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    max_results = st.number_input(
        "Max Results to Display",
        min_value=10,
        max_value=1000,
        value=100,
        step=10
    )

# Load data
try:
    with st.spinner("Loading protein interaction data..."):
        direct_interactions_df, indirect_effects_df = initialize_data()
        st.session_state.data_loaded = True
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Filter data based on user inputs
if st.session_state.data_loaded:
    active_diseases = [disease for disease, selected in disease_contexts.items() if selected]
    
    filtered_direct = get_known_interactions(
        direct_interactions_df, 
        query_protein, 
        selected_species if selected_species != "All Species" else None,
        active_diseases
    )
    
    filtered_indirect = get_known_interactions(
        indirect_effects_df,
        query_protein,
        selected_species if selected_species != "All Species" else None,
        active_diseases
    )
else:
    # Initialize empty dataframes when data is not loaded
    filtered_direct = pd.DataFrame()
    filtered_indirect = pd.DataFrame()

# Main dashboard layout
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Direct Interactions",
        len(filtered_direct) if st.session_state.data_loaded else 0
    )

with col2:
    st.metric(
        "Indirect Effects", 
        len(filtered_indirect) if st.session_state.data_loaded else 0
    )

with col3:
    if st.session_state.data_loaded and len(filtered_direct) > 0:
        avg_affinity = filtered_direct['affinity'].mean()
        avg_affinity_value = avg_affinity if not pd.isna(avg_affinity) else None
        st.metric(
            "Avg Binding Affinity",
            f"{avg_affinity_value:.2f}" if avg_affinity_value is not None else "N/A"
        )
    else:
        st.metric("Avg Binding Affinity", "N/A")

with col4:
    if st.session_state.data_loaded and len(filtered_direct) > 0:
        unique_partners = int(filtered_direct['protein'].nunique())
        st.metric("Unique Partners", unique_partners)
    else:
        st.metric("Unique Partners", 0)

# Top binding partners
if st.session_state.data_loaded and len(filtered_direct) > 0:
    st.subheader("Top 5 Binding Partners")
    top_partners = filtered_direct.nlargest(5, 'affinity')[['protein', 'gene', 'affinity', 'evidence_category']]
    st.dataframe(top_partners, use_container_width=True)

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Visualizations", "📋 Data Tables", "🔬 ML Predictions", "🧬 Protein Info", "📖 Documentation"])

with tab1:
    if st.session_state.data_loaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Binding Affinities Distribution")
            if len(filtered_direct) > 0:
                fig_bar = px.bar(
                    filtered_direct.nlargest(20, 'affinity'),
                    x='protein',
                    y='affinity',
                    color='evidence_category',
                    title="Top 20 Binding Affinities (Log Scale)",
                    log_y=True
                )
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No direct interactions found for the current query.")
        
        with col2:
            st.subheader("Evidence Categories Distribution")
            if len(filtered_direct) > 0:
                evidence_counts = filtered_direct['evidence_category'].value_counts()
                fig_pie = px.pie(
                    values=evidence_counts.values,
                    names=evidence_counts.index,
                    title="Distribution of Evidence Types"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No evidence categories to display.")
    
    # Box plot for affinity by evidence type
    if st.session_state.data_loaded and len(filtered_direct) > 0:
        st.subheader("Affinity Distribution by Evidence Type")
        fig_box = px.box(
            filtered_direct,
            x='evidence_category',
            y='affinity',
            title="Binding Affinity Distribution by Evidence Category"
        )
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Direct Interactions")
        if st.session_state.data_loaded and len(filtered_direct) > 0:
            st.dataframe(
                filtered_direct.head(max_results),
                use_container_width=True
            )
        else:
            st.info("No direct interactions found.")
        
        # Always show export buttons (even for empty results)
        if st.session_state.data_loaded:
            # Download button for direct interactions
            csv_direct = filtered_direct.to_csv(index=False)
            st.download_button(
                label="📥 Download Direct Interactions CSV",
                data=csv_direct,
                file_name=f"{query_protein or 'EGCG'}_direct_interactions.csv",
                mime="text/csv"
            )
            
            # Excel download
            excel_buffer = BytesIO()
            filtered_direct.to_excel(excel_buffer, engine='openpyxl', sheet_name='Direct_Interactions', index=False)
            excel_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Direct Interactions Excel",
                data=excel_buffer.getvalue(),
                file_name=f"{query_protein or 'EGCG'}_direct_interactions.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        st.subheader("Indirect Effects")
        if st.session_state.data_loaded and len(filtered_indirect) > 0:
            st.dataframe(
                filtered_indirect.head(max_results),
                use_container_width=True
            )
        else:
            st.info("No indirect effects found.")
        
        # Always show export buttons (even for empty results)
        if st.session_state.data_loaded:
            # Download button for indirect effects
            csv_indirect = filtered_indirect.to_csv(index=False)
            st.download_button(
                label="📥 Download Indirect Effects CSV",
                data=csv_indirect,
                file_name=f"{query_protein or 'EGCG'}_indirect_effects.csv",
                mime="text/csv"
            )
            
            # Excel download for indirect effects
            excel_buffer_indirect = BytesIO()
            filtered_indirect.to_excel(excel_buffer_indirect, engine='openpyxl', sheet_name='Indirect_Effects', index=False)
            excel_buffer_indirect.seek(0)
            
            st.download_button(
                label="📥 Download Indirect Effects Excel",
                data=excel_buffer_indirect.getvalue(),
                file_name=f"{query_protein or 'EGCG'}_indirect_effects.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

with tab3:
    st.subheader("🔬 Protein Interaction Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        protein_a = st.text_input(
            "Protein A",
            value=query_protein,
            help="Enter the first protein identifier"
        )
    
    with col2:
        protein_b = st.text_input(
            "Protein B",
            value="P53",
            help="Enter the second protein identifier"
        )
    
    if st.button("🚀 Predict Interaction", type="primary"):
        if protein_a and protein_b:
            with st.spinner("Predicting protein interaction..."):
                try:
                    prediction_result = predict_interaction(protein_a, protein_b, selected_model)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        confidence = prediction_result['confidence']
                        st.metric(
                            "Interaction Probability",
                            f"{confidence:.3f}",
                            delta=f"{confidence - confidence_threshold:.3f}"
                        )
                    
                    with col2:
                        prediction = "Likely" if confidence > confidence_threshold else "Unlikely"
                        color = "normal" if confidence > confidence_threshold else "inverse"
                        st.metric(
                            "Prediction",
                            prediction
                        )
                    
                    with col3:
                        st.metric(
                            "Model Used",
                            selected_model
                        )
                    
                    # Show interface residues if available
                    if 'interface_residues' in prediction_result and prediction_result['interface_residues']:
                        st.subheader("Predicted Interface Residues")
                        residues_df = pd.DataFrame(prediction_result['interface_residues'])
                        st.dataframe(residues_df, use_container_width=True)
                    
                    # Model explanation
                    with st.expander("Model Explanation"):
                        st.write(f"**Model Type:** {selected_model}")
                        st.write(f"**Confidence Score:** {confidence:.3f}")
                        st.write(f"**Features Used:** Protein sequence similarity, known interaction patterns, structural features")
                        if confidence > confidence_threshold:
                            st.success("The model predicts a likely interaction based on the learned patterns from known protein interactions.")
                        else:
                            st.info("The model suggests an unlikely interaction. This doesn't rule out the possibility entirely, but indicates low confidence based on current data.")
                
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
        else:
            st.warning("Please enter both protein identifiers.")
    
    # Batch prediction section
    with st.expander("Batch Prediction"):
        st.write("Upload a CSV file with protein pairs for batch prediction")
        uploaded_file = st.file_uploader(
            "Upload CSV file",
            type=['csv'],
            help="CSV should have columns 'protein_a' and 'protein_b'"
        )
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                if 'protein_a' in batch_df.columns and 'protein_b' in batch_df.columns:
                    if st.button("Run Batch Prediction"):
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, (_, row) in enumerate(batch_df.iterrows(), 1):
                            result = predict_interaction(row['protein_a'], row['protein_b'], selected_model)
                            results.append({
                                'protein_a': row['protein_a'],
                                'protein_b': row['protein_b'],
                                'confidence': result['confidence'],
                                'prediction': 'Likely' if result['confidence'] > confidence_threshold else 'Unlikely'
                            })
                            progress_bar.progress(i / len(batch_df))
                        
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Download results
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Batch Results",
                            data=csv_results,
                            file_name="batch_prediction_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("CSV must contain 'protein_a' and 'protein_b' columns")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

with tab4:
    st.subheader("🧬 Comprehensive Protein Information")
    
    # Protein lookup section
    st.write("Enter a protein identifier to retrieve comprehensive information from UniProt and PDB databases:")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        lookup_protein = st.text_input(
            "Protein Identifier",
            value=query_protein if query_protein else "P53",
            help="Enter UniProt ID, gene name, or other protein identifier"
        )
    
    with col2:
        st.write("")
        st.write("")
        lookup_button = st.button("🔍 Lookup Protein", type="primary")
    
    if lookup_button and lookup_protein:
        with st.spinner("Retrieving comprehensive protein information..."):
            try:
                protein_info = get_protein_info(lookup_protein)
                
                # Display basic information
                st.subheader(f"Basic Information for {lookup_protein.upper()}")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sequence Length", protein_info['sequence_length'])
                
                with col2:
                    if protein_info['molecular_weight'] > 0:
                        st.metric("Molecular Weight", f"{protein_info['molecular_weight']:.1f} Da")
                    else:
                        st.metric("Molecular Weight", "N/A")
                
                with col3:
                    if protein_info['isoelectric_point'] > 0:
                        st.metric("Isoelectric Point", f"{protein_info['isoelectric_point']:.2f}")
                    else:
                        st.metric("Isoelectric Point", "N/A")
                
                # UniProt metadata
                if protein_info.get('uniprot_metadata'):
                    metadata = protein_info['uniprot_metadata']
                    st.subheader("UniProt Information")
                    
                    if metadata.get('accession'):
                        st.info(f"**UniProt Accession:** {metadata['accession']}")
                    
                    if metadata.get('protein_names'):
                        st.write(f"**Protein Names:** {', '.join(metadata['protein_names'][:3])}")
                    
                    if metadata.get('gene_names'):
                        st.write(f"**Gene Names:** {', '.join(metadata['gene_names'])}")
                    
                    if metadata.get('organism'):
                        st.write(f"**Organism:** {metadata['organism']} (TaxID: {metadata.get('organism_id', 'N/A')})")
                    
                    if metadata.get('function'):
                        with st.expander("Function Description"):
                            st.write(metadata['function'][:500] + ("..." if len(metadata['function']) > 500 else ""))
                    
                    if metadata.get('subcellular_location'):
                        st.write(f"**Subcellular Location:** {', '.join(metadata['subcellular_location'][:3])}")
                    
                    if metadata.get('keywords'):
                        with st.expander("Keywords"):
                            st.write(", ".join(metadata['keywords'][:10]))
                    
                    if metadata.get('domains'):
                        with st.expander("Protein Domains"):
                            domains_df = pd.DataFrame(metadata['domains'][:5])
                            if not domains_df.empty:
                                st.dataframe(domains_df, use_container_width=True)
                
                # PDB structure information
                pdb_info = protein_info.get('pdb_structures', {})
                if pdb_info.get('has_structures'):
                    st.subheader(f"PDB Structures ({pdb_info['structures_found']} found)")
                    
                    structures = pdb_info['structures'][:5]  # Show top 5
                    structures_df = pd.DataFrame(structures)
                    if not structures_df.empty:
                        st.dataframe(structures_df, use_container_width=True)
                        
                        # AlphaFold structure link
                        alphafold_url = get_alphafold_structure_url(lookup_protein)
                        if alphafold_url:
                            st.success(f"[AlphaFold Structure Available]({alphafold_url})")
                else:
                    st.info("No PDB structures found for this protein.")
                    alphafold_url = get_alphafold_structure_url(lookup_protein)
                    if alphafold_url:
                        st.success(f"[AlphaFold Structure Available]({alphafold_url})")
                
                # Sequence display
                if protein_info.get('sequence'):
                    with st.expander("Protein Sequence"):
                        sequence = protein_info['sequence']
                        # Format sequence in blocks of 60 characters
                        formatted_seq = '\n'.join([sequence[i:i+60] for i in range(0, len(sequence), 60)])
                        st.code(formatted_seq, language=None)
                        
                        # Download sequence
                        fasta_content = f">sp|{lookup_protein.upper()}|\n{formatted_seq}"
                        st.download_button(
                            label="📥 Download FASTA",
                            data=fasta_content,
                            file_name=f"{lookup_protein.upper()}.fasta",
                            mime="text/plain"
                        )
                
                # Data source info
                st.info(f"**Data Source:** {protein_info.get('data_source', 'Unknown')}")
                
            except Exception as e:
                st.error(f"Error retrieving protein information: {str(e)}")
    
    elif lookup_protein:
        st.info("Click 'Lookup Protein' to retrieve comprehensive information from UniProt and PDB databases.")

with tab5:
    st.subheader("📖 Documentation")
    
    with st.expander("Data Sources", expanded=True):
        st.write("""
        **Direct Interactions:** Experimentally validated protein-protein interactions where two proteins physically bind to each other.
        
        **Indirect Effects:** Downstream effects where one protein influences another through intermediary proteins or pathways.
        
        **Data Sources:**
        - PubMed Central (PMC) literature mining
        - Protein interaction databases
        - Experimental validation studies
        """)
    
    with st.expander("Evidence Categories"):
        st.write("""
        - **Experimental:** Direct experimental evidence (co-immunoprecipitation, yeast two-hybrid, etc.)
        - **Computational:** Predicted interactions based on sequence/structure similarity
        - **Literature:** Interactions reported in scientific literature
        - **Database:** Curated interactions from protein databases
        """)
    
    with st.expander("Machine Learning Models"):
        st.write("""
        **Random Forest Classifier:**
        - Uses protein sequence features and known interaction patterns
        - Good interpretability and robust performance
        - Handles both numerical and categorical features
        
        **Support Vector Machine:**
        - Effective for high-dimensional protein feature spaces
        - Good generalization with limited training data
        - Uses kernel methods for complex decision boundaries
        
        **Graph Neural Network (Advanced):**
        - Models protein interaction networks as graphs
        - Captures complex relationship patterns
        - Predicts both interaction probability and interface sites
        """)
    
    with st.expander("Usage Guidelines"):
        st.write("""
        1. **Query Setup:** Enter your protein of interest and select appropriate filters
        2. **Data Exploration:** Use the visualization tabs to understand interaction patterns
        3. **Predictions:** Use the ML prediction tab for novel protein pairs
        4. **Export:** Download filtered data for further analysis
        5. **Interpretation:** Consider confidence scores and model limitations
        
        **Tips:**
        - Higher confidence scores (>0.7) indicate stronger predicted interactions
        - Cross-reference predictions with literature for validation
        - Use multiple models for consensus predictions
        """)

# Footer
st.markdown("---")
st.markdown("*Protein Interaction Dashboard - Powered by Streamlit and Machine Learning*")
