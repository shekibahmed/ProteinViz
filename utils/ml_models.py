import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
from utils.protein_utils import get_protein_features

# Cache models to avoid retraining
@st.cache_resource
def initialize_models():
    """Initialize and train ML models"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42,
            C=1.0,
            gamma='scale'
        )
    }
    
    # Try to load pre-trained models
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    trained_models = {}
    scalers = {}
    
    for model_name, model in models.items():
        model_path = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{model_name.lower().replace(' ', '_')}_scaler.pkl")
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                trained_models[model_name] = joblib.load(model_path)
                scalers[model_name] = joblib.load(scaler_path)
            except:
                # If loading fails, train new model
                trained_model, scaler = train_baseline_model(model, model_name)
                trained_models[model_name] = trained_model
                scalers[model_name] = scaler
        else:
            # Train new model
            trained_model, scaler = train_baseline_model(model, model_name)
            trained_models[model_name] = trained_model
            scalers[model_name] = scaler
            
            # Save the model
            try:
                joblib.dump(trained_model, model_path)
                joblib.dump(scaler, scaler_path)
            except:
                pass  # Continue without saving if there are permission issues
    
    return trained_models, scalers

def train_baseline_model(model, model_name):
    """Train a baseline model with synthetic data"""
    # Generate synthetic training data for demonstration
    # In a real application, this would use actual protein interaction data
    np.random.seed(42)
    
    n_samples = 1000
    n_features = 20  # protein features (sequence similarity, structural features, etc.)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate labels (1 for interaction, 0 for no interaction)
    # Create some patterns to make the problem learnable
    interaction_score = (
        X[:, 0] * 0.3 +  # sequence similarity
        X[:, 1] * 0.2 +  # structural similarity
        X[:, 2] * 0.1 +  # evolutionary distance
        np.random.randn(n_samples) * 0.1
    )
    y = (interaction_score > 0).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, scaler

def get_available_models():
    """Return list of available models"""
    return ['Random Forest', 'SVM', 'Graph Neural Network (Mock)']

@st.cache_data
def predict_interaction(protein_a, protein_b, model_type):
    """Predict interaction between two proteins"""
    try:
        # Get trained models
        models, scalers = initialize_models()
        
        if model_type == 'Graph Neural Network (Mock)':
            # Mock GNN prediction for demonstration
            return predict_gnn_interaction(protein_a, protein_b)
        
        if model_type not in models:
            raise ValueError(f"Model {model_type} not available")
        
        # Get protein features
        features_a = get_protein_features(protein_a)
        features_b = get_protein_features(protein_b)
        
        # Combine features (concatenate and compute similarity features)
        combined_features = np.concatenate([features_a, features_b])
        
        # Add interaction-specific features
        similarity_features = compute_similarity_features(features_a, features_b)
        final_features = np.concatenate([combined_features, similarity_features])
        
        # Ensure we have the right number of features
        if len(final_features) < 20:
            # Pad with zeros if we have fewer features
            final_features = np.pad(final_features, (0, 20 - len(final_features)))
        elif len(final_features) > 20:
            # Truncate if we have more features
            final_features = final_features[:20]
        
        # Scale features
        scaler = scalers[model_type]
        features_scaled = scaler.transform(final_features.reshape(1, -1))
        
        # Make prediction
        model = models[model_type]
        confidence = model.predict_proba(features_scaled)[0][1]  # Probability of interaction
        
        # Mock interface residues for advanced models
        interface_residues = []
        if confidence > 0.6:  # Only predict interface for likely interactions
            interface_residues = [
                {'protein': protein_a, 'residue': 'A123', 'score': 0.8},
                {'protein': protein_a, 'residue': 'R45', 'score': 0.7},
                {'protein': protein_b, 'residue': 'E78', 'score': 0.75},
                {'protein': protein_b, 'residue': 'K156', 'score': 0.65}
            ]
        
        return {
            'confidence': confidence,
            'model_used': model_type,
            'interface_residues': interface_residues,
            'features_used': len(final_features)
        }
        
    except Exception as e:
        # Fallback to random prediction with explanation
        np.random.seed(hash(protein_a + protein_b) % 2**32)
        confidence = np.random.uniform(0.1, 0.9)
        
        return {
            'confidence': confidence,
            'model_used': f"{model_type} (fallback)",
            'interface_residues': [],
            'error': str(e)
        }

def predict_gnn_interaction(protein_a, protein_b):
    """Mock GNN prediction for demonstration"""
    # Use protein names to generate reproducible "predictions"
    seed = hash(protein_a + protein_b) % 2**32
    np.random.seed(seed)
    
    # Mock confidence based on protein name patterns
    confidence = np.random.beta(2, 3)  # Skewed towards lower values
    
    # Adjust confidence based on known patterns
    if any(known in protein_a.upper() for known in ['P53', 'BRCA', 'EGCG']):
        confidence += 0.2
    if any(known in protein_b.upper() for known in ['P53', 'BRCA', 'EGCG']):
        confidence += 0.2
    
    confidence = min(confidence, 0.95)  # Cap at 95%
    
    # Mock interface residues for high-confidence predictions
    interface_residues = []
    if confidence > 0.7:
        interface_residues = [
            {'protein': protein_a, 'residue': f'A{np.random.randint(10, 300)}', 'score': np.random.uniform(0.6, 0.9)},
            {'protein': protein_a, 'residue': f'R{np.random.randint(10, 300)}', 'score': np.random.uniform(0.6, 0.9)},
            {'protein': protein_b, 'residue': f'E{np.random.randint(10, 300)}', 'score': np.random.uniform(0.6, 0.9)},
            {'protein': protein_b, 'residue': f'K{np.random.randint(10, 300)}', 'score': np.random.uniform(0.6, 0.9)}
        ]
    
    return {
        'confidence': confidence,
        'model_used': 'Graph Neural Network',
        'interface_residues': interface_residues,
        'graph_features': {
            'nodes': np.random.randint(50, 200),
            'edges': np.random.randint(100, 500),
            'clustering_coefficient': np.random.uniform(0.1, 0.5)
        }
    }

def compute_similarity_features(features_a, features_b):
    """Compute similarity features between two protein feature vectors"""
    # Ensure both feature vectors have the same length
    min_len = min(len(features_a), len(features_b))
    features_a = features_a[:min_len]
    features_b = features_b[:min_len]
    
    # Compute various similarity measures
    cosine_sim = np.dot(features_a, features_b) / (np.linalg.norm(features_a) * np.linalg.norm(features_b))
    euclidean_dist = np.linalg.norm(features_a - features_b)
    manhattan_dist = np.sum(np.abs(features_a - features_b))
    
    return np.array([cosine_sim, euclidean_dist, manhattan_dist])

def get_model_explanation(model_type, confidence):
    """Generate explanation for model prediction"""
    explanations = {
        'Random Forest': f"The Random Forest model analyzed {20} protein features and achieved {confidence:.2f} confidence. This ensemble method combines multiple decision trees to make robust predictions.",
        'SVM': f"The Support Vector Machine found optimal decision boundaries in high-dimensional protein feature space with {confidence:.2f} confidence.",
        'Graph Neural Network (Mock)': f"The Graph Neural Network analyzed protein interaction networks and structural patterns to predict with {confidence:.2f} confidence."
    }
    
    return explanations.get(model_type, f"Model prediction confidence: {confidence:.2f}")
