# Ongoing research

# Introduction

This technical report discusses various approaches to generating sentence embeddings, which are vector representations of text that capture the semantic and contextual meaning of sentences. The goal of sentence embeddings is to represent sentences in a compact, fixed-length format that can be used for a variety of natural language processing tasks, such as text classification, information retrieval, and semantic similarity analysis.

# Literature Review

The report covers several common techniques for generating sentence embeddings:

- **Dense Bag of Words (Word Frequency Count):** This simple approach counts the frequency of each word in a sentence, resulting in a vector representation where each element corresponds to the frequency of a specific word. The order of words in the sentence is not considered.
- **TF-IDF (Term Frequency-Inverse Document Frequency):** This method evaluates the importance of a word within a document relative to a collection of documents (corpus). It emphasizes terms that are both frequent within a document and unique or rare across the entire corpus, allowing for the representation of documents as vectors in a high-dimensional space.
- **Doc2Vec:** An extension of Word2Vec, Doc2Vec learns fixed-length embeddings for documents (sentences, paragraphs, or entire documents) by training alongside word vectors, effectively capturing sentence-level semantics.
- **Global Average Pooling:** Before the advent of more sophisticated sentence embedding approaches, it was common to average the word embeddings from models like Word2Vec, GloVe, or BERT to produce a dense sentence embedding.
- **LDA Topic Modeling:** Latent Dirichlet Allocation (LDA) can be used to model topics in a document or sentence, and the resulting topic proportions can be used as embeddings.
- **Siamese Networks & Contrastive Pre-training:** Siamese networks are used for similarity-based tasks, where two identical networks with shared parameters are trained to produce similar embeddings for semantically related sentences.
- **Last Hidden State of a Bidirectional RNN/GRU/LSTM:** Recurrent neural networks can capture contextual relations in a sequence, and the concatenation or averaging of the last hidden states from forward and backward RNNs can be used as sentence embeddings.
- **Conv1D + Average Pooling (Semi-Original Solution):** Convolutional neural networks can capture contextual information over longer sequences more efficiently than RNNs, and the output of a 1D convolution layer followed by average pooling can be used as sentence embeddings.
- **Gaussian Mixture of Models (Original Solution):** In this approach, a Gaussian Mixture Model is used to assign each sentence vector to a cluster, and the probabilities of the sentence belonging to each cluster are used as the sentence embedding.

# Warping Contrastive Loss

A sentence embedding is a vector in space. Imagine this vector as a multidimensional lump, when we scale an element of the vector, this lump is stretched out or pressed in that direction. If we continously scale every element by different values, this can be visualised as the lump warping around a region. Effectively, this warping covers possible sentences close to the point as it touch on surrounding regions.

Warped Embeddings = Embedding * (1 + N(0.3, 0.2)) 

Given two sets of input embeddings, $a$ and $b$, for a batch size of $N$, the combined loss function, $\text{combined\_loss}(a, b)$, incorporates the steps of normalization, warping, similarity matrix computation, softmax probability conversion, and cross-entropy calculation as follows:

1. **Warp and Normalize Input Embeddings:**
   $$
   A_i = \frac{a_i}{\|a_i\|_2} \odot (S_{a_i} + \mathbf{1}), \quad B_i = \frac{b_i}{\|b_i\|_2} \odot (S_{b_i} + \mathbf{1})
   $$

   where $S_{a_i}$ and kGiven two sets of input embeddings, $a$ and $b$, for a batch size of $N$, the combined loss function, $\text{combined\_loss}(a, b)$, incorporates the steps of normalization, warping, similarity matrix computation, softmax probability conversion, and cross-entropy calculation as follows:

1. **Warp and Normalize Input Embeddings:**
   For each input tensor $a_i$ and $b_i$ in the batches $a$ and $b$:

   $$
   A_i = \frac{a_i}{\|a_i\|_2} \odot (S_{a_i} + \mathbf{1}), \quad B_i = \frac{b_i}{\|b_i\|_2} \odot (S_{b_i} + \mathbf{1})
   $$

   where $S_{a_i}$ and $S_{b_i}$ are samples from normal distributions parameterized by mean $\mu$ and standard deviation $\sigma$ specific to each element of $a_i$ and $b_i$.

2. **Compute Similarity Matrix and Apply Softmax:**
   The similarity matrix $M$ is calculated as:

   $$
   M = \frac{AB^\top}{\tau}
   $$

   where each element $M_{ij}$ represents the scaled cosine similarity between $A_i$ and $B_j$. The softmax function is then applied to each row of $M$ to convert these similarities into probabilities:

   $$
   P_{ij} = \frac{e^{M_{ij}}}{\sum_{k=1}^{N} e^{M_{ik}}}
   $$

3. **Compute Cross-Entropy Loss:**
   The cross-entropy loss for each pair of actual and predicted distributions is calculated as follows:

   $$
   \text{loss}_a = -\frac{1}{N} \sum_{i=1}^{N} \log(P_{ii})
   $$

   $$
   \text{loss}_b = -\frac{1}{N} \sum_{i=1}^{N} \log(P_{ii}^\top)
   $$

   Here, $P_{ii}^\top$ refers to the probability that the i-th element of $b$ correctly matches the i-th element of $a$ after transpose, effectively capturing the b-to-a direction.

4. **Combined Loss Expression:**
   The final combined loss merges the two directional losses:

   $$
   \text{combined\_loss}(a, b) = \frac{\text{loss}_a + \text{loss}_b}{2}
   $$

   $$
   = -\frac{1}{2N} \left( \sum_{i=1}^{N} \log\left(\frac{e^{M_{ii}}}{\sum_{k=1}^{N} e^{M_{ik}}}\right) + \sum_{i=1}^{N} \log\left(\frac{e^{M_{ii}^\top}}{\sum_{k=1}^{N} e^{M_{ki}^\top}}\right) \right)
   $$

This formal expression encapsulates the entire process: starting from the input embeddings, applying normalization and warping, constructing the similarity matrix, converting similarities to probabilities with softmax, and finally calculating the average cross-entropy loss from both directions of similarity.
$S_{b_i}$ are samples from normal distributions parameterized by mean $\mu$ and standard deviation $\sigma$ specific to each element of $a_i$ and $b_i$.

2. **Compute Similarity Matrix and Apply Softmax:**
   The similarity matrix $M$ is calculated as:

   $$
   M = \frac{AB^\top}{\tau}
   $$

   where each element $M_{ij}$ represents the scaled cosine similarity between $A_i$ and $B_j$. The softmax function is then applied to each row of $M$ to convert these similarities into probabilities:

   $$
   P_{ij} = \frac{e^{M_{ij}}}{\sum_{k=1}^{N} e^{M_{ik}}}
   $$

3. **Compute Cross-Entropy Loss:**
   The cross-entropy loss for each pair of actual and predicted distributions is calculated as follows:

   $$
   \text{loss}_a = -\frac{1}{N} \sum_{i=1}^{N} \log(P_{ii})
   $$

   $$
   \text{loss}_b = -\frac{1}{N} \sum_{i=1}^{N} \log(P_{ii}^\top)
   $$

   Here, $P_{ii}^\top$ refers to the probability that the i-th element of $b$ correctly matches the i-th element of $a$ after transpose, effectively capturing the b-to-a direction.

4. **Combined Loss Expression:**
   The final combined loss merges the two directional losses:

   $$
   \text{combined\_loss}(a, b) = \frac{\text{loss}_a + \text{loss}_b}{2}
   $$

   $$
   = -\frac{1}{2N} \left( \sum_{i=1}^{N} \log\left(\frac{e^{M_{ii}}}{\sum_{k=1}^{N} e^{M_{ik}}}\right) + \sum_{i=1}^{N} \log\left(\frac{e^{M_{ii}^\top}}{\sum_{k=1}^{N} e^{M_{ki}^\top}}\right) \right)
   $$

This formal expression encapsulates the entire process: starting from the input embeddings, applying normalization and warping, constructing the similarity matrix, converting similarities to probabilities with softmax, and finally calculating the average cross-entropy loss from both directions of similarity.


# Results and Discussion

So far, this novel approach significantly outperformed both the CLIP-Contrastive (+12% Pearson correlation) and NLI (+36.5% Pearson correlation) techniques under the same conditions. The report also discusses potential improvements, such as explicitly enforcing orthogonality between dissimilar embeddings and designing the model to avoid representations with colinear elements, to further enhance the performance of the sentence embeddings.