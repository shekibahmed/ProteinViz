"""3D Protein Structure Visualization Module.

This module provides a comprehensive set of functions for fetching,
displaying, and styling 3D protein structures within a Streamlit
application. It leverages `stmol` and `py3Dmol` for interactive
visualizations and supports various representations, color schemes, and
highlighting of specific residues. It is designed to handle structures from
both the RCSB PDB and the AlphaFold Database.
"""

import streamlit as st
import py3Dmol
from stmol import showmol, render_pdb, render_pdb_resn, speck_plot
import numpy as np
import requests
from typing import Dict, List, Optional, Tuple
import pandas as pd
# Bio.PDB imports (commenting out as they're not currently used)
# from Bio.PDB.PDBParser import PDBParser
# from Bio.PDB.PDBIO import PDBIO
from io import StringIO


@st.cache_data(ttl=86400)  # Extended caching: 24 hours
def fetch_pdb_structure(pdb_id: str) -> Optional[str]:
    """Fetch a PDB structure file from the RCSB PDB database.

    This function downloads the PDB file for a given PDB ID. It includes
    error handling for network issues and invalid IDs. The results are
    cached for 24 hours to improve performance.

    Args:
        pdb_id (str): The 4-character PDB identifier.

    Returns:
        str, optional: The content of the PDB file as a string, or None if
            the file cannot be fetched.
    """
    try:
        pdb_id = pdb_id.upper().strip()
        if len(pdb_id) != 4:
            return None
            
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.text
        else:
            st.warning(f"Could not fetch PDB structure {pdb_id} from RCSB PDB")
            return None
            
    except Exception as e:
        st.warning(f"Error fetching PDB {pdb_id}: {str(e)}")
        return None

@st.cache_data(ttl=86400)  # Extended caching: 24 hours
def fetch_alphafold_structure(uniprot_id: str) -> Optional[str]:
    """Fetch a predicted protein structure from the AlphaFold Database.

    This function downloads a PDB file of a protein structure predicted by
    AlphaFold, based on a UniProt ID. The results are cached for 24 hours.

    Args:
        uniprot_id (str): The UniProt identifier for the protein.

    Returns:
        str, optional: The content of the PDB file as a string, or None if
            not found.
    """
    try:
        uniprot_id = uniprot_id.upper().strip()
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            return response.text
        else:
            st.info(f"No AlphaFold structure found for {uniprot_id}")
            return None
            
    except Exception as e:
        st.warning(f"Error fetching AlphaFold structure for {uniprot_id}: {str(e)}")
        return None

def create_3d_viewer(pdb_data: str, width: int = 800, height: int = 500) -> py3Dmol.view:
    """Create a basic 3D molecular viewer with PDB data.

    This function initializes a `py3Dmol` viewer instance and loads the
    protein structure data into it.

    Args:
        pdb_data (str): A string containing the protein structure in PDB
            format.
        width (int, optional): The width of the viewer in pixels. Defaults
            to 800.
        height (int, optional): The height of the viewer in pixels.
            Defaults to 500.

    Returns:
        py3Dmol.view: An instance of the py3Dmol viewer.
    """
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_data, 'pdb')
    return viewer

def style_protein_cartoon(viewer: py3Dmol.view, color_scheme: str = 'spectrum') -> py3Dmol.view:
    """Apply a cartoon representation to a protein structure.

    This function styles the protein in the viewer with a cartoon
    representation, which is useful for visualizing secondary structures.
    It supports various color schemes.

    Args:
        viewer (py3Dmol.view): The py3Dmol viewer instance.
        color_scheme (str, optional): The color scheme to apply. Options
            include 'spectrum', 'secondary', 'hydrophobicity', 'chain', or a
            specific color name. Defaults to 'spectrum'.

    Returns:
        py3Dmol.view: The viewer instance with the applied style.
    """
    if color_scheme == 'spectrum':
        viewer.setStyle({'cartoon': {'color': 'spectrum'}})
    elif color_scheme == 'secondary':
        viewer.setStyle({'cartoon': {'colorscheme': 'sstruc'}})
    elif color_scheme == 'hydrophobicity':
        viewer.setStyle({'cartoon': {'colorscheme': 'hydrophobicity'}})
    elif color_scheme == 'chain':
        viewer.setStyle({'cartoon': {'colorscheme': 'chain'}})
    else:
        viewer.setStyle({'cartoon': {'color': color_scheme}})
    
    return viewer

def style_protein_surface(viewer: py3Dmol.view, opacity: float = 0.7) -> py3Dmol.view:
    """Add a surface representation to a protein structure.

    This function adds a molecular surface to the protein in the viewer,
    which can be rendered with a specified opacity.

    Args:
        viewer (py3Dmol.view): The py3Dmol viewer instance.
        opacity (float, optional): The opacity of the surface. Defaults to 0.7.

    Returns:
        py3Dmol.view: The viewer instance with the added surface style.
    """
    viewer.addStyle({'surface': {'opacity': opacity, 'color': 'white'}})
    return viewer

def highlight_residues(viewer: py3Dmol.view, residues: Optional[List[Dict]] = None, style: str = 'sphere') -> py3Dmol.view:
    """Highlight specific residues in the structure.

    This function applies a distinct style to a list of specified residues,
    making them stand out in the visualization. This is useful for marking
    active sites or interaction interfaces.

    Args:
        viewer (py3Dmol.view): The py3Dmol viewer instance.
        residues (list[dict], optional): A list of dictionaries, where each
            dictionary contains information about a residue to highlight,
            including its number and a score for coloring. Defaults to None.
        style (str, optional): The style to use for highlighting ('sphere'
            or 'stick'). Defaults to 'sphere'.

    Returns:
        py3Dmol.view: The viewer instance with the highlighted residues.
    """
    if not residues:
        return viewer
    
    for res_info in residues:
        try:
            # Extract residue number from residue string (e.g., 'A123' -> 123)
            residue_str = res_info.get('residue', '')
            if residue_str and len(residue_str) > 1:
                # Handle both 'A123' and '123' formats
                residue_num = ''.join(filter(str.isdigit, residue_str))
                if residue_num:
                    residue_num = int(residue_num)
                    
                    # Color based on score if available
                    score = res_info.get('score', 0.5)
                    if score > 0.8:
                        color = 'red'
                    elif score > 0.6:
                        color = 'orange'
                    else:
                        color = 'yellow'
                    
                    # Add highlighting style
                    selection = {'resi': residue_num}
                    if style == 'sphere':
                        viewer.addStyle(selection, {'sphere': {'color': color, 'radius': 1.0}})
                    elif style == 'stick':
                        viewer.addStyle(selection, {'stick': {'color': color, 'radius': 0.3}})
                    else:
                        viewer.addStyle(selection, {'cartoon': {'color': color}})
        except Exception as e:
            continue  # Skip problematic residues
    
    return viewer

def create_network_visualization(proteins: List[str], interactions: List[Tuple[str, str, float]]) -> py3Dmol.view:
    """Create a 3D network visualization of protein interactions.

    This function generates a 3D representation of a protein interaction
    network. Proteins are represented as spheres, and interactions are shown
    as cylinders connecting them.

    Args:
        proteins (list[str]): A list of protein identifiers.
        interactions (list[tuple[str, str, float]]): A list of tuples, each
            representing an interaction with the two protein identifiers and
            a confidence score.

    Returns:
        py3Dmol.view: A py3Dmol viewer instance containing the 3D network.
    """
    viewer = py3Dmol.view(width=800, height=600)
    
    # Position proteins in 3D space
    n_proteins = len(proteins)
    positions = {}
    
    # Create a spherical layout
    for i, protein in enumerate(proteins):
        theta = 2 * np.pi * i / n_proteins
        phi = np.pi * (i + 0.5) / n_proteins
        
        x = 10 * np.sin(phi) * np.cos(theta)
        y = 10 * np.sin(phi) * np.sin(theta)
        z = 10 * np.cos(phi)
        
        positions[protein] = (x, y, z)
        
        # Add protein as a sphere
        viewer.addSphere({
            'center': {'x': x, 'y': y, 'z': z},
            'radius': 2.0,
            'color': 'blue',
            'alpha': 0.8
        })
        
        # Add protein label
        viewer.addLabel(protein, {
            'position': {'x': x, 'y': y + 3, 'z': z},
            'backgroundColor': 'white',
            'fontColor': 'black'
        })
    
    # Add interaction lines
    for protein1, protein2, confidence in interactions:
        if protein1 in positions and protein2 in positions:
            pos1 = positions[protein1]
            pos2 = positions[protein2]
            
            # Line color based on confidence
            if confidence > 0.8:
                color = 'red'
                radius = 0.3
            elif confidence > 0.6:
                color = 'orange'
                radius = 0.2
            else:
                color = 'gray'
                radius = 0.1
            
            viewer.addCylinder({
                'start': {'x': pos1[0], 'y': pos1[1], 'z': pos1[2]},
                'end': {'x': pos2[0], 'y': pos2[1], 'z': pos2[2]},
                'radius': radius,
                'color': color,
                'alpha': 0.7
            })
    
    viewer.zoomTo()
    return viewer

def display_protein_structure_tab(protein_id: str, pdb_data: str, interface_residues: Optional[List[Dict]] = None):
    """Display a comprehensive 3D protein structure visualization.

    This function creates a Streamlit tab with a 3D viewer for a protein
    structure. It includes controls for changing the color scheme and
    representation, and for highlighting interface residues.

    Args:
        protein_id (str): The identifier of the protein being displayed.
        pdb_data (str): The protein structure data in PDB format.
        interface_residues (list[dict], optional): A list of interface
            residues to highlight. Defaults to None.
    """
    
    st.subheader(f"3D Structure Viewer - {protein_id}")
    
    # Visualization controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color_scheme = st.selectbox(
            "Color Scheme",
            ["spectrum", "secondary", "hydrophobicity", "chain", "blue", "red", "green"],
            key=f"color_{protein_id}"
        )
    
    with col2:
        representation = st.selectbox(
            "Representation",
            ["cartoon", "stick", "sphere", "surface", "cartoon+surface"],
            key=f"repr_{protein_id}"
        )
    
    with col3:
        show_interface = st.checkbox(
            "Highlight Interface Residues",
            value=True if interface_residues else False,
            key=f"interface_{protein_id}"
        )
    
    with col4:
        viewer_height = st.slider(
            "Viewer Height",
            min_value=300,
            max_value=800,
            value=500,
            step=50,
            key=f"height_{protein_id}"
        )
    
    try:
        # Create 3D viewer
        viewer = create_3d_viewer(pdb_data, height=viewer_height)
        
        # Apply representation styles
        if representation == "cartoon":
            viewer = style_protein_cartoon(viewer, color_scheme)
        elif representation == "stick":
            viewer.setStyle({'stick': {'color': color_scheme if color_scheme in ['blue', 'red', 'green'] else 'spectrum'}})
        elif representation == "sphere":
            viewer.setStyle({'sphere': {'color': color_scheme if color_scheme in ['blue', 'red', 'green'] else 'spectrum'}})
        elif representation == "surface":
            viewer.setStyle({'surface': {'color': color_scheme if color_scheme in ['blue', 'red', 'green'] else 'white', 'opacity': 0.8}})
        elif representation == "cartoon+surface":
            viewer = style_protein_cartoon(viewer, color_scheme)
            viewer = style_protein_surface(viewer, opacity=0.3)
        
        # Highlight interface residues if available
        if show_interface and interface_residues:
            viewer = highlight_residues(viewer, interface_residues, style='sphere')
            
            # Show interface residues information
            with st.expander(f"🎯 Interface Residues ({len(interface_residues)} found)"):
                interface_df = pd.DataFrame(interface_residues)
                st.dataframe(interface_df, use_container_width=True)
        
        # Set up viewer
        viewer.setBackgroundColor('white')
        viewer.zoomTo()
        
        # Display the 3D viewer
        showmol(viewer, height=viewer_height, width=800)
        
        # Add structure information
        with st.expander("📊 Structure Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Visualization Settings:**")
                st.write(f"- Color Scheme: {color_scheme}")
                st.write(f"- Representation: {representation}")
                st.write(f"- Interface Residues: {'Highlighted' if show_interface and interface_residues else 'None'}")
            
            with col2:
                st.write("**Structure Details:**")
                st.write(f"- Protein ID: {protein_id}")
                st.write(f"- Data Source: {'PDB' if pdb_data.startswith('HEADER') else 'AlphaFold'}")
                if interface_residues:
                    st.write(f"- Interface Sites: {len(interface_residues)}")
    
    except Exception as e:
        st.error(f"Error displaying 3D structure: {str(e)}")
        st.info("The structure viewer encountered an issue. Please try a different protein or check the PDB ID.")

def display_interaction_network(protein_pairs: List[Tuple[str, str]], predictions: List[Dict]):
    """Display a 3D protein interaction network.

    This function creates a Streamlit component for visualizing a protein
    interaction network in 3D. It includes controls for filtering
    interactions by a confidence threshold.

    Args:
        protein_pairs (list[tuple[str, str]]): A list of protein pairs.
        predictions (list[dict]): A list of prediction dictionaries, each
            containing a confidence score.
    """
    
    st.subheader("🕸️ 3D Protein Interaction Network")
    
    if not protein_pairs or not predictions:
        st.info("No protein interactions available for network visualization.")
        return
    
    # Extract unique proteins
    all_proteins = set()
    for pair in protein_pairs:
        all_proteins.update(pair)
    proteins = list(all_proteins)
    
    # Extract interactions with confidence scores
    interactions = []
    for i, (protein1, protein2) in enumerate(protein_pairs):
        if i < len(predictions):
            confidence = predictions[i].get('confidence', 0.5)
            interactions.append((protein1, protein2, confidence))
    
    # Network controls
    col1, col2 = st.columns(2)
    
    with col1:
        min_confidence = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        network_height = st.slider(
            "Network Height",
            min_value=400,
            max_value=800,
            value=600,
            step=50
        )
    
    # Filter interactions by confidence
    filtered_interactions = [(p1, p2, conf) for p1, p2, conf in interactions if conf >= min_confidence]
    
    if not filtered_interactions:
        st.warning(f"No interactions found above confidence threshold {min_confidence}")
        return
    
    try:
        # Create network visualization
        network_viewer = create_network_visualization(proteins, filtered_interactions)
        
        # Display the network
        showmol(network_viewer, height=network_height, width=800)
        
        # Network statistics
        with st.expander("📈 Network Statistics"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Proteins", len(proteins))
            
            with col2:
                st.metric("Total Interactions", len(filtered_interactions))
            
            with col3:
                avg_confidence = np.mean([conf for _, _, conf in filtered_interactions]) if filtered_interactions else 0
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            # Interaction details
            if filtered_interactions:
                st.write("**Interaction Details:**")
                interaction_data = [{'Protein A': p1, 'Protein B': p2, 'Confidence': conf} for p1, p2, conf in filtered_interactions]
                network_df = pd.DataFrame(interaction_data)
                st.dataframe(network_df.sort_values('Confidence', ascending=False), use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying interaction network: {str(e)}")
        st.info("The network visualization encountered an issue. Please try adjusting the parameters.")

def get_structure_for_protein(protein_id: str) -> Optional[str]:
    """Get protein structure data from PDB or AlphaFold.

    This function attempts to find and fetch protein structure data, first by
    treating the ID as a PDB ID, and then as a UniProt ID for AlphaFold.

    Args:
        protein_id (str): The identifier for the protein.

    Returns:
        str, optional: A string containing the PDB-formatted structure data,
            or None if no structure is found.
    """
    
    # First try as PDB ID (4 characters)
    if len(protein_id) == 4:
        pdb_data = fetch_pdb_structure(protein_id)
        if pdb_data:
            return pdb_data
    
    # Try as UniProt ID for AlphaFold
    alphafold_data = fetch_alphafold_structure(protein_id)
    if alphafold_data:
        return alphafold_data
    
    # If both fail, try variations
    if len(protein_id) > 4:
        # Try first 4 characters as PDB ID
        pdb_data = fetch_pdb_structure(protein_id[:4])
        if pdb_data:
            return pdb_data
    
    return None

def display_structure_comparison(protein_a: str, protein_b: str, interface_residues_a: Optional[List[Dict]] = None, interface_residues_b: Optional[List[Dict]] = None):
    """Display a side-by-side comparison of two protein structures.

    This function creates a two-column layout in Streamlit to display the
    3D structures of two different proteins next to each other, allowing for
    easy visual comparison.

    Args:
        protein_a (str): The identifier for the first protein.
        protein_b (str): The identifier for the second protein.
        interface_residues_a (list[dict], optional): Interface residues for
            the first protein. Defaults to None.
        interface_residues_b (list[dict], optional): Interface residues for
            the second protein. Defaults to None.
    """
    
    st.subheader("🔄 Protein Structure Comparison")
    
    # Get structures
    structure_a = get_structure_for_protein(protein_a)
    structure_b = get_structure_for_protein(protein_b)
    
    if not structure_a and not structure_b:
        st.warning(f"No 3D structures available for {protein_a} or {protein_b}")
        return
    
    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{protein_a}**")
        if structure_a:
            display_protein_structure_tab(protein_a, structure_a, interface_residues_a)
        else:
            st.info(f"No structure available for {protein_a}")
    
    with col2:
        st.write(f"**{protein_b}**")
        if structure_b:
            display_protein_structure_tab(protein_b, structure_b, interface_residues_b)
        else:
            st.info(f"No structure available for {protein_b}")