# Import Fireplace modules
from fireplace import cards

# Import other libraries
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch

# Load configuration
import yaml
from yaml.loader import SafeLoader
    
# Function to load the sentence transformer model
def load_model(model_name: str = 'all-MiniLM-L6-v2') -> SentenceTransformer:
    # Make docstring with rst syntax
    '''
    Load the SentenceTransformer model.\n
    \n
    Parameters:\n
    - model_name: The name of the model to load\n
    \n
    Returns:\n
    - model: The SentenceTransformer model
    '''
    
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load the model
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    return model

# Function to embed a card name
def embed_card_name(card_name: str, model: SentenceTransformer) -> np.ndarray:
    
    """
    
    Function to embed a card name using the SentenceTransformer model
    
    :param card_name: str
    :param pipeline: SentenceEmbeddingPipeline
    :return: np.ndarray
    
    """    
    # Convert to numpy
    np_embedding = model.encode(card_name)
    
    return np_embedding

if __name__ == '__main__':
    
    with open('embedding_config.yaml') as config_file:
        embed_config = yaml.load(config_file, Loader=SafeLoader)
    
    # Load the cards database
    cards.db.initialize()
    
    card_tuples = [(card.id, card.name, card.classes[0].name) for card in cards.db.values()]

    # Write to a csv file
    card_df = pd.DataFrame(card_tuples, columns=['id', 'name', 'class'])
    card_df.to_csv("cards.csv", index=False)
    
    # Load the card embedding model
    model = load_model()
    
    # Create large data frame with card ids, names, classes, types, and descriptions
    card_data = pd.DataFrame(columns=['id', 'id_hash', 'name', 'class', 'type', 'description'])
    for card in tqdm(cards.db.values()):
        card_data = card_data._append({
            'id': card.id, 
            'id_hash': hash(card.id), 
            'name': card.name, 
            'class': card.classes[0].name, 
            'type': card.type, 
            'description': card.description
            }, ignore_index=True)
        
    # Create a np array with the embeddings and card ids
    embeddings = []
    card_ids = []
    for card_name in tqdm(card_data['name']):
        card_ids.append(card_data[card_data['name'] == card_name]['id'].values[0])
        embeddings.append(embed_card_name(card_name, model))
        
    embeddings = np.array(embeddings)
    card_ids = np.array(card_ids)
    
    # Save the embeddings and card ids as npz files
    np.savez('card_embeddings.npz', embeddings=embeddings, card_ids=card_ids)
    
    