# -*- coding: utf-8 -*-
"""
Hyperspectral Unmixing with Multi-Scale GAT and Inter-Superpixel PCR

This implementation performs hyperspectral unmixing using:
- Multi-Scale Graph Attention Networks (GAT) for feature extraction
- Local-Global Attention Fusion (LGAF) for feature fusion
- Inter-Superpixel PCR for abundance smoothing
- Adaptive Class-Dependent Endmember (ACDE) for endmember learning

Original work based on MSGAA-CD with Inter-Superpixel PCR.
"""

from scipy.ndimage import gaussian_filter
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io
import os

# Configure TensorFlow for optimal performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU available, using CPU")

# Enable mixed precision for faster training (if GPU available)
if gpus:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    print("Mixed precision enabled")

# Set TensorFlow to use multiple CPU threads
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
print(f"CPU threads configured")


# Load the Samson HSI data and Endmembers
samson_data = scipy.io.loadmat('data/samson.mat')
endmember_data = scipy.io.loadmat('data/end3.mat')

# Extract shape information
nRow = int(samson_data['nRow'][0][0])
nCol = int(samson_data['nCol'][0][0])
nBand = int(samson_data['nBand'][0][0])


def create_graph_from_superpixels(image, segments, k=5):
    num_segments = segments.max() + 1
    H, W, C = image.shape

    # Vectorized feature computation
    segments_flat = segments.flatten()
    image_flat = image.reshape(-1, C)

    # Use advanced indexing for faster computation
    features = np.zeros((num_segments, C), dtype=np.float32)
    counts = np.zeros(num_segments, dtype=np.float32)

    for seg_id in range(num_segments):
        mask = segments_flat == seg_id
        if np.any(mask):
            features[seg_id] = np.mean(image_flat[mask], axis=0)
            counts[seg_id] = np.sum(mask)

    # Spatial adjacency - vectorized
    adj = np.zeros((num_segments, num_segments), dtype=np.float32)

    # Horizontal edges
    h_edges = np.zeros((H, W-1), dtype=np.int32)
    h_edges[:, :] = segments[:, :-1]
    h_edges_right = segments[:, 1:]
    mask_h = h_edges != h_edges_right
    s1_h = h_edges[mask_h]
    s2_h = h_edges_right[mask_h]
    adj[s1_h, s2_h] = 1
    adj[s2_h, s1_h] = 1

    # Vertical edges
    v_edges = np.zeros((H-1, W), dtype=np.int32)
    v_edges[:, :] = segments[:-1, :]
    v_edges_bottom = segments[1:, :]
    mask_v = v_edges != v_edges_bottom
    s1_v = v_edges[mask_v]
    s2_v = v_edges_bottom[mask_v]
    adj[s1_v, s2_v] = 1
    adj[s2_v, s1_v] = 1

    # Cosine similarity-based edges (top-k neighbors) - vectorized
    sim_matrix = cosine_similarity(features)
    np.fill_diagonal(sim_matrix, -np.inf)  # Exclude self-similarity

    for i in range(num_segments):
        top_k = np.argsort(-sim_matrix[i])[:k]
        adj[i, top_k] = 1
        adj[top_k, i] = 1  # symmetric

    return features.astype(np.float32), adj, segments


class MultiScaleGATEncoder(tf.keras.layers.Layer):
    def __init__(self, num_endmembers=3, num_heads=1):
        super().__init__()
        self.num_endmembers = num_endmembers
        self.num_heads = num_heads
        self.superpixel_scales = [2000, 1000, 500]

        input_dim = 156
        out_features = num_endmembers
        self.gat_layers = [
            [
                CustomGATConv(input_dim, 64),
                CustomGATConv(64, out_features)
            ] for _ in range(len(self.superpixel_scales))
        ]

        self.relu = tf.keras.layers.ReLU()
        self.softmax = tf.keras.layers.Softmax(axis=-1)

        self.scale_weights = self.add_weight(
            shape=(len(self.superpixel_scales),),
            initializer='random_uniform',
            trainable=True,
            name='scale_weights'
        )

    def call(self, Y):
        # If input is a flat pixel vector: [1, 156]
        if Y.shape.ndims == 2 and Y.shape[0] == 1:
            x = Y
            x = self.gat_layers[0][0]([x, tf.eye(tf.shape(x)[0])])
            x = self.gat_layers[0][1]([x, tf.eye(tf.shape(x)[0])])
            x = self.relu(x)
            x = self.softmax(x)
            return x  # [1, E]

        # Else, if input is full HSI image: [H, W, C]
        height = tf.shape(Y)[0]
        width = tf.shape(Y)[1]
        channels = tf.shape(Y)[2]

        # Convert TensorFlow tensor to NumPy array for SLIC
        # Handle both eager and graph modes
        try:
            Y_np = Y.numpy()
        except (AttributeError, TypeError):
            # In graph mode, use K.get_value or convert via tf.py_function
            Y_np = tf.keras.backend.get_value(Y) if hasattr(
                tf.keras.backend, 'get_value') else tf.make_ndarray(tf.make_tensor_proto(Y))

        S_scales = []

        for i, n_seg in enumerate(self.superpixel_scales):
            segments = slic(Y_np, n_segments=n_seg,
                            compactness=10, start_label=0)
            feats, adj, _ = create_graph_from_superpixels(Y_np, segments)
            X = tf.convert_to_tensor(feats, dtype=tf.float32)
            A = tf.convert_to_tensor(adj, dtype=tf.float32)

            x = self.gat_layers[i][0]([X, A])
            x = self.gat_layers[i][1]([x, A])
            x = self.relu(x)
            x = self.softmax(x)

            # Use TensorFlow operations to map superpixel features back to pixels
            # This preserves gradient flow
            segments_tf = tf.convert_to_tensor(segments, dtype=tf.int32)
            segments_flat_tf = tf.reshape(segments_tf, [-1])

            # Use tf.gather to map features: for each pixel, get its superpixel's feature
            S_tau_flat = tf.gather(x, segments_flat_tf)
            S_tau = tf.reshape(
                S_tau_flat, [height, width, self.num_endmembers])
            S_scales.append(S_tau)

        stacked = tf.stack(S_scales, axis=0)
        w = tf.nn.softmax(self.scale_weights)
        w = tf.reshape(w, (-1, 1, 1, 1))
        fused = tf.reduce_sum(stacked * w, axis=0)

        return fused  # [H, W, E]


class CNNLocalEncoder(tf.keras.layers.Layer):
    def __init__(self, num_abundances=3):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            64, 3, padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(
            128, 3, padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(
            num_abundances, 3, padding='same', activation='softmax')

    def call(self, Y):
        # Y must be either [H, W, C] or [B, H, W, C]
        if tf.rank(Y) == 3:
            Y = tf.expand_dims(Y, axis=0)         # -> [1, H, W, C]
        elif tf.rank(Y) != 4:
            raise ValueError(
                f"Unexpected rank {tf.rank(Y)} for CNNLocalEncoder input")

        x = self.conv1(Y)                         # [B, H, W, 64]
        x = self.conv2(x)                         # [B, H, W, 128]
        x = self.conv3(x)                         # [B, H, W, num_abundances]
        return x                                  # [B, H, W, num_abundances]


class LGAF(tf.keras.layers.Layer):
    def __init__(self, num_endmembers=3):
        super(LGAF, self).__init__()
        self.num_endmembers = num_endmembers

        self.linear_global = tf.keras.layers.Conv2D(
            num_endmembers, 1, padding='same', kernel_initializer='he_normal')
        self.linear_local = tf.keras.layers.Conv2D(
            num_endmembers, 1, padding='same', kernel_initializer='he_normal')

        self.alpha = tf.Variable(1.0, trainable=True, dtype=tf.float32)
        self.beta = tf.Variable(1.0, trainable=True, dtype=tf.float32)

        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def match_shape(self, SG, SC):
        h1, w1 = tf.shape(SG)[0], tf.shape(SG)[1]
        h2, w2 = tf.shape(SC)[0], tf.shape(SC)[1]

        min_h = tf.minimum(h1, h2)
        min_w = tf.minimum(w1, w2)

        SG = SG[:min_h, :min_w, :]
        SC = SC[:min_h, :min_w, :]

        return SG, SC

    def call(self, SG, SC):  # Both: [H, W, C] for a single image
        SG, SC = self.match_shape(SG, SC)

        # Add batch dimension for Conv2D layers
        SG_batch = tf.expand_dims(SG, axis=0)  # [1, H, W, C]
        SC_batch = tf.expand_dims(SC, axis=0)  # [1, H, W, C]

        # Concat on channel dimension
        FG_input = tf.concat([SG_batch, SC_batch], axis=-1)  # [1, H, W, 2C]
        # [1, H, W, num_endmembers]
        FG = self.sigmoid(self.linear_global(FG_input))

        # Remove batch dimension for element-wise operations
        FG = tf.squeeze(FG, axis=0)  # [H, W, num_endmembers]
        SC_hat = self.alpha * FG * SG + SC

        # Re-add batch dimension for Conv2D
        FC_input = tf.concat([tf.expand_dims(SC, axis=0), tf.expand_dims(
            SG, axis=0)], axis=-1)  # [1, H, W, 2C]
        # [1, H, W, num_endmembers]
        FC = self.sigmoid(self.linear_local(FC_input))

        # Remove batch dimension for element-wise operations
        FC = tf.squeeze(FC, axis=0)  # [H, W, num_endmembers]
        SG_hat = self.beta * FC * SC + SG

        S = self.softmax(SG_hat + SC_hat)  # [H, W, num_endmembers]
        return S


class InterSuperpixelPCR(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, fused_abundances, superpixel_segments=None, attention_matrix=None, alpha=None):
        H = tf.shape(fused_abundances)[0]
        W = tf.shape(fused_abundances)[1]
        E = tf.shape(fused_abundances)[2]

        if superpixel_segments is None:
            # Create dummy superpixels (avoid .numpy())
            segments_flat = tf.range(H * W) % 1000
            superpixel_segments = tf.reshape(segments_flat, [H, W])

        num_superpixels = tf.reduce_max(superpixel_segments) + 1
        fused_flat = tf.reshape(fused_abundances, [-1, E])
        segments_flat = tf.cast(tf.reshape(
            superpixel_segments, [-1]), tf.int32)

        superpixel_features = tf.math.unsorted_segment_mean(
            fused_flat, segments_flat, num_superpixels)

        if attention_matrix is None:
            attention_matrix = tf.eye(num_superpixels, dtype=tf.float32)

        attention_weights = tf.nn.softmax(attention_matrix, axis=-1)
        smoothed_features = tf.matmul(attention_weights, superpixel_features)

        smoothed_abundances = tf.gather(smoothed_features, segments_flat)
        smoothed_abundances = tf.reshape(smoothed_abundances, [H, W, E])

        if alpha is None:
            alpha = 0.5
        alpha_clipped = tf.clip_by_value(alpha, 0.0, 1.0)
        corrected_abundances = alpha_clipped * smoothed_abundances + \
            (1 - alpha_clipped) * fused_abundances

        return corrected_abundances


class ACDE(tf.keras.layers.Layer):
    def __init__(self,
                 num_endmembers,    # P = 3
                 num_features,      # C = 156
                 hidden_dim=128,
                 name="ACDE"):
        super().__init__(name=name)
        self.num_endmembers = num_endmembers
        self.num_features = num_features

        # Define the MLP network for learning weights for endmembers
        self.mlp = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(num_features)
        ])

    def call(self, abundance_matrix, Y):
        # abundance_matrix: [N1, A]
        # Y:                 [N2, C]

        N_Y = tf.shape(Y)[0]  # Number of pixels in Y (H * W)
        N_S = tf.shape(abundance_matrix)[0]  # Number of pixels in S (H * W)

        # Ensure both have the same number of pixels (this step is no longer necessary in a single image)
        if N_Y != N_S:
            raise ValueError(
                "The number of pixels in the abundance matrix and the hyperspectral image must be the same.")

        # Flatten for processing: [target_N, ...]
        Y_flat = tf.reshape(Y, [N_Y, self.num_features])             # [N_Y, C]
        S_flat = tf.reshape(abundance_matrix, [
                            N_S, self.num_endmembers])  # [N_S, A]

        # Compute class indices
        class_indices = tf.argmax(S_flat, axis=-1)  # [N_Y]

        # Compute endmembers
        endmembers = []
        for p in range(self.num_endmembers):
            mask = tf.equal(class_indices, p)               # [N_Y]
            class_pixels = tf.boolean_mask(Y_flat, mask)    # [? , C]
            num_class_pixels = tf.shape(class_pixels)[0]

            def no_pixels():
                return tf.zeros((self.num_features,), dtype=Y_flat.dtype)

            def some_pixels():
                w = tf.nn.softmax(self.mlp(class_pixels), axis=0)  # [? , C]
                return tf.reduce_sum(w * class_pixels, axis=0)     # [C]

            endmember = tf.cond(num_class_pixels > 0, some_pixels, no_pixels)
            endmembers.append(endmember)

        M = tf.stack(endmembers, axis=0)  # [A, C]

        # Reconstruct
        # S_flat @ M  => [N_Y, C]
        Y_hat_flat = tf.matmul(S_flat, M)
        # reshape back to [N_Y, C]
        Y_hat = tf.reshape(Y_hat_flat, [N_Y, self.num_features])

        return Y_hat


def compute_lsad(Y, Y_hat):
    # Y and Y_hat: [N, C] where N is the number of pixels, and C is the number of features
    Y_norm = tf.norm(Y, axis=1)  # [N]
    Y_hat_norm = tf.norm(Y_hat, axis=1)  # [N]
    dot_product = tf.reduce_sum(Y * Y_hat, axis=1)  # [N]

    eps = 1e-8
    cosine = tf.clip_by_value(
        dot_product / (Y_norm * Y_hat_norm + eps), -1.0, 1.0)
    angles = tf.acos(cosine)  # [N]
    return tf.reduce_mean(angles)  # Scalar


def compute_lmse(Y, Y_hat):
    # Y and Y_hat: [N, C]
    # Scalar
    return tf.reduce_mean(tf.reduce_sum(tf.square(Y - Y_hat), axis=1))


def compute_total_loss(Y, Y_hat, abundance_matrix, lambda1=1.0, lambda2=0.001, q=0.5):
    """
    Final loss = LSAD + λ1 * LMSE + λ2 * ||S||_q (sparsity constraint on abundance)
    """
    # Calculate the LSAD and LMSE losses
    lsad = compute_lsad(Y, Y_hat)
    lmse = compute_lmse(Y, Y_hat)

    # Lq-norm sparsity on the abundance matrix
    eps = 1e-8
    lq_norm = tf.reduce_mean(
        tf.pow(tf.abs(abundance_matrix) + eps, q))  # Sparsity constraint

    # Total loss
    total_loss = lsad + lambda1 * lmse + lambda2 * lq_norm
    return total_loss, lsad, lmse, lq_norm


class CustomGATConv(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.attn_kernel = self.add_weight(shape=(input_dim, output_dim),
                                           initializer='glorot_uniform',
                                           trainable=True)
        self.attn_bias = self.add_weight(shape=(output_dim,),
                                         initializer='zeros',
                                         trainable=True)

    def call(self, inputs):
        X, A = inputs  # X: [N, C], A: [N, N]

        H = tf.matmul(X, self.attn_kernel) + self.attn_bias  # [N, output_dim]
        attention = tf.matmul(H, tf.transpose(H, perm=[1, 0]))  # [N, N]
        attention = tf.where(A > 0, attention, tf.fill(
            tf.shape(attention), -1e9))  # Apply adjacency mask
        attention_weights = tf.nn.softmax(attention, axis=-1)  # [N, N]

        output = tf.matmul(attention_weights, H)  # [N, output_dim]
        return tf.nn.relu(output)  # Apply ReLU activation


# Endmembers shape: (156, 3)
# Abundances shape: (3, 9025)

# Extract hyperspectral data and reshape
V = samson_data['V']  # [nBand, nRow*nCol]
Y = V.T.reshape(nRow, nCol, nBand)  # Reshape to [nRow, nCol, nBand]

# Extract ground truth endmembers and abundances
M_true = endmember_data['M']  # [nBand, num_endmembers]
A_true = endmember_data['A']  # [num_endmembers, nRow*nCol]

num_endmembers = M_true.shape[1]  # Should be 3 for Samson
print(f"Image shape: {Y.shape}")
print(f"Endmembers shape: {M_true.shape}")
print(f"Abundances shape: {A_true.shape}")

# Reshape ground truth abundances for evaluation
A_true_reshaped = np.zeros((nRow, nCol, num_endmembers))
for i in range(num_endmembers):
    A_true_reshaped[:, :, i] = A_true[i, :].reshape(nRow, nCol)

# Normalize the hyperspectral data to [0, 1] range
Y_min = np.min(Y)
Y_max = np.max(Y)
Y_normalized = (Y - Y_min) / (Y_max - Y_min)


class HyperspectralUnmixingModel(tf.keras.Model):
    def __init__(self, num_endmembers, num_bands, initial_alpha=0.5):
        super(HyperspectralUnmixingModel, self).__init__()
        self.encoder = MultiScaleGATEncoder(num_endmembers=num_endmembers)
        self.fusion = LGAF(num_endmembers=num_endmembers)
        self.alpha = tf.Variable(
            initial_value=initial_alpha, trainable=True, dtype=tf.float32, name='alpha')
        self.pcr = InterSuperpixelPCR()
        self.decoder = ACDE(num_endmembers=num_endmembers,
                            num_features=num_bands)

    def call(self, inputs, training=False, superpixel_segments=None, attention_matrix=None):
        Z = self.encoder(inputs, training=training)
        Z_fused = self.fusion(Z, Z, training=training)
        S_pcr = self.pcr(Z_fused, superpixel_segments=superpixel_segments,
                         attention_matrix=attention_matrix, alpha=self.alpha)

        # Flatten inputs and S_pcr for ACDE
        H, W = tf.shape(inputs)[0], tf.shape(inputs)[1]
        inputs_flat = tf.reshape(inputs, [H * W, tf.shape(inputs)[2]])
        S_pcr_flat = tf.reshape(S_pcr, [H * W, tf.shape(S_pcr)[2]])
        Y_hat_flat = self.decoder(S_pcr_flat, inputs_flat)
        Y_hat = tf.reshape(Y_hat_flat, [H, W, tf.shape(inputs)[2]])

        return S_pcr, Z_fused, Y_hat


def train_step(model, optimizer, Y, superpixel_segments=None, attention_matrix=None,
               lambda1=1.0, lambda2=0.001, q=0.5):
    with tf.GradientTape() as tape:
        S_smoothed, S_fused, Y_hat = model(Y, superpixel_segments=superpixel_segments,
                                           attention_matrix=attention_matrix, training=True)

        H, W = tf.shape(Y)[0], tf.shape(Y)[1]
        Y_flat = tf.reshape(Y, [H * W, tf.shape(Y)[2]])
        Y_hat_flat = tf.reshape(Y_hat, [H * W, tf.shape(Y_hat)[2]])
        S_flat = tf.reshape(S_smoothed, [H * W, tf.shape(S_smoothed)[2]])

        total_loss, lsad, lmse, lq_norm = compute_total_loss(
            Y_flat, Y_hat_flat, S_flat, lambda1=lambda1, lambda2=lambda2, q=q
        )

    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return total_loss, lsad, lmse, lq_norm, S_smoothed, S_fused


model = HyperspectralUnmixingModel(
    num_endmembers=num_endmembers, num_bands=nBand, initial_alpha=0.5)
optimizer = Adam(learning_rate=0.001)
Y_tf = tf.convert_to_tensor(Y_normalized, dtype=tf.float32)

epochs = 200
losses = []
fused_history = []
smoothed_history = []
final_fused = None
final_smoothed = None
display_step = 10

print("Starting training...")

for epoch in range(epochs):
    total_loss, lsad, lmse, lq_norm, S_smoothed, S_fused = train_step(
        model, optimizer, Y_tf, lambda1=1.0, lambda2=0.001, q=0.5
    )

    losses.append({'total': total_loss.numpy(),
                   'sad': lsad.numpy(),
                   'mse': lmse.numpy(),
                   'lq': lq_norm.numpy()})

    if epoch % 10 == 0:
        fused_history.append(S_fused.numpy())
        smoothed_history.append(S_smoothed.numpy())

    if epoch == epochs - 1:
        final_fused = S_fused
        final_smoothed = S_smoothed

    if epoch % display_step == 0:
        print("Alpha (clipped):", tf.clip_by_value(
            model.alpha, 0.0, 1.0).numpy())
        print(f"Epoch {epoch}: Total Loss = {total_loss.numpy():.4f}, "
              f"SAD = {lsad.numpy():.4f}, MSE = {lmse.numpy():.4f}, Lq = {lq_norm.numpy():.4f}")

print("Training completed!")

final_fused_np = final_fused.numpy()
final_smoothed_np = final_smoothed.numpy()

estimated_endmembers = model.decoder(
    tf.reshape(final_smoothed, [-1, num_endmembers]),
    tf.reshape(Y_tf, [-1, nBand])
)
estimated_endmembers = tf.transpose(estimated_endmembers, perm=[1, 0])
estimated_endmembers = estimated_endmembers.numpy()

# -------------------------
# Plotting
# -------------------------

# 1. Training Losses
plt.figure(figsize=(10, 6))
plt.plot([l['total'] for l in losses], label='Total Loss')
plt.plot([l['sad'] for l in losses], label='SAD Loss')
plt.plot([l['mse'] for l in losses], label='MSE Loss')
plt.title('Training Losses')
plt.legend()
plt.savefig('results/training_losses.png')
plt.show()

# 2. Abundance Maps Comparison (Fused vs PCR-smoothed vs Ground Truth)
plt.figure(figsize=(18, 12))
for i in range(num_endmembers):
    # Fused abundance maps
    plt.subplot(3, num_endmembers, i + 1)
    plt.imshow(final_fused_np[:, :, i])
    plt.colorbar()
    plt.title(f'Fused Abundance {i+1}')

    # PCR-smoothed abundance maps
    plt.subplot(3, num_endmembers, i + 1 + num_endmembers)
    plt.imshow(final_smoothed_np[:, :, i])
    plt.colorbar()
    plt.title(f'PCR-Smoothed Abundance {i+1}')

    # Ground truth
    plt.subplot(3, num_endmembers, i + 1 + 2*num_endmembers)
    plt.imshow(A_true_reshaped[:, :, i])
    plt.colorbar()
    plt.title(f'Ground Truth Abundance {i+1}')

plt.tight_layout()
plt.savefig('results/abundance_comparison.png')
plt.show()

# 3. Endmembers Comparison
plt.figure(figsize=(10, 6))
for i in range(num_endmembers):
    plt.plot(range(nBand),
             estimated_endmembers[:, i], label=f'Estimated Endmember {i+1}')
    plt.plot(range(nBand), M_true[:, i], '--', label=f'True Endmember {i+1}')
plt.title('Endmembers Comparison')
plt.legend()
plt.savefig('results/endmembers_comparison.png')
plt.show()
