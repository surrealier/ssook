# Embedding & CLIP

[← Back to Index](index.md) | 🌐 [한국어](ko/embedding-and-clip.md) | [日本語](ja/embedding-and-clip.md) | [中文](zh/embedding-and-clip.md)

ssook supports embedding visualization, CLIP zero-shot classification, and embedder evaluation. This document explains the underlying principles and how to use each feature.

---

## Table of Contents

- [Embedding Visualization](#embedding-visualization)
  - [t-SNE](#t-sne)
  - [UMAP](#umap)
  - [PCA](#pca)
- [CLIP Zero-Shot Classification](#clip-zero-shot-classification)
- [Embedder Evaluation](#embedder-evaluation)

---

## Embedding Visualization

Embedding models produce high-dimensional vectors (e.g., 512 or 768 dimensions) for each image. To visualize how images cluster, ssook reduces these vectors to 2D using one of three algorithms.

### t-SNE

**t-Distributed Stochastic Neighbor Embedding**

**How it works:**
1. In the original high-dimensional space, compute pairwise similarities using a Gaussian distribution. Nearby points get high similarity; distant points get low similarity. The **perplexity** parameter (typically 5–50) controls how many neighbors each point considers.
2. In the 2D target space, compute pairwise similarities using a Student's t-distribution (heavier tails than Gaussian).
3. Minimize the **KL divergence** (a measure of difference) between the two similarity distributions using gradient descent.
4. The t-distribution's heavy tails prevent the "crowding problem" — in 2D, moderately distant points aren't forced to collapse together.

**Characteristics:**
- ✅ Excellent at revealing local cluster structure
- ✅ Visually clear separation between classes
- ⚠️ Global distances are not meaningful — the distance between clusters doesn't reflect actual similarity
- ⚠️ Non-deterministic — different runs may produce different layouts
- ⚠️ Slow for large datasets (O(n²) complexity)

### UMAP

**Uniform Manifold Approximation and Projection**

**How it works:**
1. Build a weighted k-nearest-neighbor graph in the high-dimensional space. Each point connects to its nearest neighbors, with edge weights based on distance.
2. Construct a similar graph in the 2D target space.
3. Optimize the 2D layout to make the two graphs as similar as possible, using cross-entropy loss.
4. The mathematical foundation is topological — UMAP preserves the "shape" (manifold structure) of the data.

**Characteristics:**
- ✅ Preserves both local and global structure better than t-SNE
- ✅ Faster than t-SNE, scales to larger datasets
- ✅ More deterministic across runs
- ⚠️ Requires the `umap-learn` package

### PCA

**Principal Component Analysis**

**How it works:**
1. Center the data by subtracting the mean of each dimension.
2. Compute the covariance matrix of the centered data.
3. Find the eigenvectors (principal components) of the covariance matrix. The first eigenvector points in the direction of maximum variance, the second in the direction of maximum remaining variance (orthogonal to the first), and so on.
4. Project the data onto the top 2 eigenvectors to get 2D coordinates.

**Characteristics:**
- ✅ Deterministic — same input always produces same output
- ✅ Very fast (linear algebra, no iteration)
- ✅ Global structure is preserved — distances are meaningful
- ⚠️ Only captures linear relationships — may miss complex cluster structures
- ⚠️ If variance is spread across many dimensions, 2D projection may look like a blob

**How to use embedding visualization in ssook:**
1. Go to **Analysis** tab → **Embedding Visualization**
2. Load any ONNX feature extractor model
3. Set the image directory (folder structure: `class_name/image.jpg`)
4. Select the algorithm (t-SNE / UMAP / PCA)
5. Click **Run** — a 2D scatter plot is generated with points colored by class

---

## CLIP Zero-Shot Classification

CLIP (Contrastive Language-Image Pre-training) enables classifying images using text descriptions, without any task-specific training data.

**How it works:**

1. **Dual encoder architecture**: CLIP consists of two separate neural networks:
   - **Image encoder**: Converts an image into a fixed-size embedding vector
   - **Text encoder**: Converts a text description into an embedding vector of the same size

2. **Image preprocessing**: The input image is resized to the encoder's expected resolution (typically 224×224), converted to RGB, normalized using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), and arranged as a NCHW tensor.

3. **Zero-shot classification flow**:
   - Encode the input image → image embedding
   - Encode each candidate text label (e.g., "a photo of a cat", "a photo of a dog") → text embeddings
   - Compute **cosine similarity** between the image embedding and each text embedding:
     ```
     similarity = (image_emb · text_emb) / (‖image_emb‖ × ‖text_emb‖)
     ```
   - The text label with the highest similarity is the predicted class

4. **Why it works**: During pre-training, CLIP learned to align images and their text descriptions in a shared embedding space. Images of cats are close to the text "a photo of a cat" in this space.

**How to use in ssook:**
1. Go to **CLIP** tab
2. Load the image encoder ONNX model and text encoder ONNX model
3. Enter candidate text labels (one per line)
4. Load an image or set of images
5. Click **Run** — similarity scores for each label are displayed

---

## Embedder Evaluation

Embedder evaluation measures how well a feature extraction model separates different classes in embedding space.

**How it works:**

1. **Dataset structure**: Images organized as `class_name/image.jpg`. Each folder name is a class label.
2. **Embedding extraction**: Each image is passed through the ONNX model to produce an embedding vector.
3. **L2 normalization**: All embeddings are normalized to unit length, so cosine similarity equals the dot product.
4. **Similarity matrix**: Compute the full Q×G cosine similarity matrix between query and gallery embeddings.
5. **Metrics**:
   - **Retrieval@1**: For each query, is the top-1 most similar gallery image from the same class?
   - **Retrieval@K**: For each query, does the correct class appear in the top-K results?
   - **Mean cosine similarity**: Average of the maximum similarity for each query (how confident the matches are)

**Multi-image comparison**: You can also select multiple images and view a pairwise cosine similarity matrix, useful for checking if specific images are similar or different in embedding space.

**How to use in ssook:**
1. Go to **Evaluation** tab → **Embedder**
2. Load the feature extractor ONNX model
3. Set the dataset directory
4. Set K for Retrieval@K (default: 5)
5. Click **Run**
