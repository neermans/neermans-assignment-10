from flask import Flask, request, render_template
import os
import numpy as np
import torch
from PIL import Image
from open_clip import get_tokenizer, create_model_and_transforms
import pandas as pd
import torch.nn.functional as F
from sklearn.decomposition import PCA

app = Flask(__name__)

# Load model, tokenizer, and preprocess
model, _, preprocess = create_model_and_transforms('ViT-B/32', pretrained='openai')
tokenizer = get_tokenizer('ViT-B-32')
model.eval()

# Load precomputed embeddings and file names
df = pd.read_pickle('image_embeddings.pickle')
embeddings = np.vstack(df['embedding'].values)

# Prepend the directory path to each file name to construct the full paths
file_names = [os.path.join('static/coco_images_resized', fname) for fname in df['file_name'].values]

# PCA setup for optional dimensionality reduction
pca = PCA(n_components=10)
pca_embeddings = pca.fit_transform(embeddings)

# Helper functions
def compute_similarity(query_embedding, embeddings):
    cosine_similarities = np.dot(embeddings, query_embedding.T).squeeze()
    top_indices = np.argsort(cosine_similarities)[::-1][:5]
    return [(file_names[idx], cosine_similarities[idx]) for idx in top_indices]

def process_text_query(text):
    text_token = tokenizer([text])
    text_embedding = F.normalize(model.encode_text(text_token)).detach().cpu().numpy()
    return text_embedding

def process_image_query(image_file):
    image = preprocess(Image.open(image_file)).unsqueeze(0)
    image_embedding = F.normalize(model.encode_image(image)).detach().cpu().numpy()
    return image_embedding

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        query_type = request.form.get('query_type')
        weight = float(request.form.get('weight', 0.5))  # Weight for text query (lam)
        use_pca = request.form.get('use_pca') == 'on'

        query_embedding = None

        # Handle text query
        if query_type == 'text':
            text = request.form.get('text_query')
            query_embedding = process_text_query(text)
            if use_pca:
                query_embedding = pca.transform(query_embedding)

        # Handle image query
        elif query_type == 'image':
            image_file = request.files['image_query']
            query_embedding = process_image_query(image_file)
            if use_pca:
                query_embedding = pca.transform(query_embedding)

        # Handle combined query
        elif query_type == 'both':
            text = request.form.get('text_query')
            image_file = request.files['image_query']

            # Process text and image queries
            text_embedding = process_text_query(text)
            image_embedding = process_image_query(image_file)

            # Combine embeddings using the tunable parameter `weight`
            text_embedding_tensor = torch.tensor(text_embedding, dtype=torch.float32)
            image_embedding_tensor = torch.tensor(image_embedding, dtype=torch.float32)
            combined_embedding = F.normalize(weight * text_embedding_tensor + (1.0 - weight) * image_embedding_tensor, p=2, dim=1)

            # Convert back to NumPy for compatibility
            combined_embedding_numpy = combined_embedding.detach().cpu().numpy()

            # Apply PCA if selected
            if use_pca:
                query_embedding = pca.transform(combined_embedding_numpy)
            else:
                query_embedding = combined_embedding_numpy

        # Find top 5 matches
        results = compute_similarity(query_embedding.squeeze(), pca_embeddings if use_pca else embeddings)

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)