import os
import math
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from sklearn.metrics.pairwise import cosine_similarity

# Check for PyTorch Geometric (PyG) availability
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data, Batch
    PYG_AVAILABLE = True
except ImportError:
    print("Warning: torch_geometric not found. Using simple fallback GAT implementation.")
    PYG_AVAILABLE = False

# ==========================================
# 1. Differentiable SLIC (Superpixel) Module
# ==========================================


class DiffSLIC(nn.Module):
    """
    Differentiable Superpixel Linear Iterative Clustering.
    Computes soft assignment Q between pixels and superpixel centers.
    """

    def __init__(self, num_features, num_superpixels=100, compactness=10.0, num_iterations=5):
        super().__init__()
        self.num_superpixels = num_superpixels
        self.compactness = compactness
        self.num_iterations = num_iterations
        self.num_features = num_features

    def initialize_centers(self, x, H, W):
        # Grid initialization
        grid_h = int(math.sqrt(self.num_superpixels * H / W))
        grid_w = self.num_superpixels // grid_h

        h_step = H // grid_h
        w_step = W // grid_w

        h_indices = torch.arange(
            h_step // 2, H, h_step, device=x.device)[:grid_h]
        w_indices = torch.arange(
            w_step // 2, W, w_step, device=x.device)[:grid_w]

        grid_y, grid_x = torch.meshgrid(h_indices, w_indices, indexing='ij')
        init_centers_spatial = torch.stack(
            [grid_y.flatten(), grid_x.flatten()], dim=1).float()

        # Sample spectral features at these locations
        flat_indices = (grid_y.flatten() * W + grid_x.flatten()).long()
        x_flat = x.reshape(-1, self.num_features)
        init_centers_spectral = x_flat[flat_indices]

        return init_centers_spectral, init_centers_spatial

    def forward(self, x):
        """
        Args:
            x: Input image [B, C, H, W]
        Returns:
            Q: Soft assignment matrix [B, H*W, K]
            super_feats: Superpixel features [B, K, C]
        """
        B, C, H, W = x.shape
        device = x.device

        # Create pixel coordinate grid
        y_grid, x_grid = torch.meshgrid(torch.arange(
            H, device=device), torch.arange(W, device=device), indexing='ij')
        pixel_coords = torch.stack(
            [y_grid.flatten(), x_grid.flatten()], dim=1).float()  # [N, 2]

        x_flat = x.permute(0, 2, 3, 1).reshape(B, -1, C)  # [B, N, C]

        # Initialize centers (simplified for batch: use first image to init)
        # In a full impl, we might track centers as learnable params or state
        centers_spectral, centers_spatial = self.initialize_centers(x[0], H, W)
        centers_spectral = centers_spectral.unsqueeze(
            0).expand(B, -1, -1)  # [B, K, C]
        centers_spatial = centers_spatial.unsqueeze(
            0).expand(B, -1, -1)   # [B, K, 2]

        Q = None

        for _ in range(self.num_iterations):
            # Calculate distances
            # Spectral distance: ||c_s - p_s||^2
            # Spatial distance: ||c_xy - p_xy||^2

            # Efficient implementation needed for N*K distances.
            # We assume K is small (~100-200) and N is large (~20000).

            d_spectral = torch.cdist(x_flat, centers_spectral)  # [B, N, K]
            d_spatial = torch.cdist(pixel_coords.unsqueeze(
                0).expand(B, -1, -1), centers_spatial)  # [B, N, K]

            # Combined distance (S = m/S * d_xy)
            S = math.sqrt(H * W / self.num_superpixels)
            D = d_spectral + (self.compactness / S) * d_spatial

            # Soft assignment (Softmax over K)
            # Using negative distance as logit
            Q = F.softmax(-D, dim=-1)  # [B, N, K]

            # Update centers
            # Weighted sum of pixel features / sum of weights
            mass = torch.sum(Q, dim=1, keepdim=True) + 1e-6  # [B, 1, K]

            centers_spectral = torch.bmm(Q.transpose(
                1, 2), x_flat) / mass.transpose(1, 2)

            # Update spatial centers
            pixel_coords_batch = pixel_coords.unsqueeze(
                0).expand(B, -1, -1)  # [B, N, 2]
            centers_spatial = torch.bmm(Q.transpose(
                1, 2), pixel_coords_batch) / mass.transpose(1, 2)  # [B, K, 2]

        return Q, centers_spectral

# ==========================================
# 2. Graph Attention Module (MSGAA)
# ==========================================


class SimpleGATLayer(nn.Module):
    """Fallback if PyG is not available"""

    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2*out_features, 1, bias=False)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = self.W(h)  # [B, K, F]
        B, K, _ = Wh.size()

        # Attention mechanism (Simplified global attention for demo)
        # In practice, use sparse operations or PyG
        a_input = torch.cat([Wh.repeat(1, 1, K).view(B, K*K, -1),
                             Wh.repeat(1, K, 1)], dim=2).view(B, K, K, -1)
        e = self.leakyrelu(self.a(a_input).squeeze(3))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = torch.bmm(attention, Wh)
        return F.elu(h_prime)


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        if PYG_AVAILABLE:
            self.conv1 = GATConv(in_channels, hidden_dim,
                                 heads=num_heads, concat=True)
            self.conv2 = GATConv(hidden_dim * num_heads,
                                 out_channels, heads=1, concat=False)
        else:
            self.conv1 = SimpleGATLayer(in_channels, hidden_dim)
            self.conv2 = SimpleGATLayer(hidden_dim, out_channels)

    def forward(self, x, adj_matrix_or_edge_index):
        if PYG_AVAILABLE:
            # x: [K, C], edge_index: [2, E]
            x = F.elu(self.conv1(x, adj_matrix_or_edge_index))
            x = self.conv2(x, adj_matrix_or_edge_index)
        else:
            # x: [B, K, C], adj: [B, K, K]
            x = self.conv1(x, adj_matrix_or_edge_index)
            x = self.conv2(x, adj_matrix_or_edge_index)

        return x

# ==========================================
# 3. CNN Local Encoder
# ==========================================


class CNNEncoder(nn.Module):
    def __init__(self, in_channels, num_endmembers):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_endmembers, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 4. Cross Attention Fusion (LGAF)
# ==========================================


class CrossAttentionFusion(nn.Module):
    def __init__(self, num_endmembers):
        super().__init__()
        self.linear_global = nn.Conv2d(
            num_endmembers * 2, num_endmembers, 1, padding=0)
        self.linear_local = nn.Conv2d(
            num_endmembers * 2, num_endmembers, 1, padding=0)
        self.alpha = nn.Parameter(torch.tensor(1.5))
        self.beta = nn.Parameter(torch.tensor(0.3))

    def forward(self, SG, SC):
        """
        SG: Global abundances [B, P, H, W]
        SC: Local abundances [B, P, H, W]
        """
        # Match shapes if needed
        B, P, H1, W1 = SG.shape
        _, _, H2, W2 = SC.shape
        H = min(H1, H2)
        W = min(W1, W2)
        SG = SG[:, :, :H, :W]
        SC = SC[:, :, :H, :W]

        # Global branch: SG queries SC
        FG_input = torch.cat([SG, SC], dim=1)  # [B, 2P, H, W]
        FG = torch.sigmoid(self.linear_global(FG_input))  # [B, P, H, W]
        SC_hat = self.alpha * FG * SG + SC

        # Local branch: SC queries SG
        FC_input = torch.cat([SC, SG], dim=1)  # [B, 2P, H, W]
        FC = torch.sigmoid(self.linear_local(FC_input))  # [B, P, H, W]
        SG_hat = self.beta * FC * SC + SG

        # Final fusion - use softmax for proper probability distribution
        S = F.softmax(SG_hat + SC_hat, dim=1)
        return S

# ==========================================
# 4.5. VCA Initialization
# ==========================================


def vca_initialization(Y, num_endmembers):
    """
    Vertex Component Analysis for endmember initialization.
    Y: [H*W, C] or [H, W, C]
    Returns: [C, P] endmembers
    """
    if len(Y.shape) == 3:
        Y_flat = Y.reshape(-1, Y.shape[-1])
    else:
        Y_flat = Y

    Y_flat = Y_flat - Y_flat.mean(axis=0, keepdims=True)

    # Project to num_endmembers-1 dimensional space
    U, S, Vt = np.linalg.svd(Y_flat.T, full_matrices=False)
    U_reduced = U[:, :num_endmembers-1]

    # Project data
    Y_proj = Y_flat @ U_reduced

    # Initialize with extreme points
    endmembers = np.zeros((Y_flat.shape[1], num_endmembers))
    endmembers[:, 0] = Y_flat[np.argmax(np.linalg.norm(Y_proj, axis=1))]

    for p in range(1, num_endmembers):
        # Find point with maximum distance from current simplex
        distances = np.zeros(Y_proj.shape[0])
        for i in range(Y_proj.shape[0]):
            # Distance to current simplex
            simplex = endmembers[:, :p].T @ U_reduced
            dist = np.min(np.linalg.norm(Y_proj[i:i+1] - simplex, axis=1))
            distances[i] = dist

        endmembers[:, p] = Y_flat[np.argmax(distances)]

    return endmembers.astype(np.float32)


# ==========================================
# 5. ACDE Decoder (Physics-based)
# ==========================================


class ACDEDecoder(nn.Module):
    def __init__(self, num_endmembers, num_bands, hidden_dim=128, init_M=None):
        super().__init__()
        self.num_endmembers = num_endmembers
        self.num_bands = num_bands

        # Learnable endmember matrix M [C, P] with VCA initialization
        if init_M is not None:
            self.M = nn.Parameter(torch.from_numpy(init_M).float())
        else:
            self.M = nn.Parameter(torch.rand(num_bands, num_endmembers))

    def forward(self, abundances, Y):
        """
        abundances: [B, P, H, W]
        Y: [B, C, H, W] - original image
        Returns: reconstructed [B, C, H, W], endmembers [C, P]
        """
        B, P, H, W = abundances.shape
        N = H * W

        # Ensure M is non-negative and L2 normalized (preserves spectral contrast)
        M_constrained = F.relu(self.M)
        M_constrained = M_constrained / \
            (torch.norm(M_constrained, p=2, dim=0, keepdim=True) + 1e-8)

        # Flatten for reconstruction
        A_flat = abundances.permute(0, 2, 3, 1).reshape(B * N, P)  # [B*N, P]

        # Reconstruction: Y_hat = A @ M^T
        Y_hat_flat = torch.matmul(A_flat, M_constrained.t())  # [B*N, C]
        Y_hat = Y_hat_flat.reshape(B, H, W, self.num_bands).permute(
            0, 3, 1, 2)  # [B, C, H, W]

        return Y_hat, M_constrained

# ==========================================
# 5.5. Inter-Superpixel PCR
# ==========================================


class InterSuperpixelPCR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fused_abundances, superpixel_segments=None, attention_matrix=None, alpha=None):
        """
        fused_abundances: [B, P, H, W]
        superpixel_segments: [H, W] numpy array or None
        attention_matrix: [K, K] or None
        alpha: float tensor or None
        """
        B, P, H, W = fused_abundances.shape
        device = fused_abundances.device

        if superpixel_segments is None:
            # Create dummy superpixels
            segments_flat = torch.arange(H * W, device=device) % 1000
            segments = segments_flat.reshape(H, W).cpu().numpy()
        else:
            segments = superpixel_segments

        num_superpixels = int(segments.max()) + 1
        fused_flat = fused_abundances.permute(
            0, 2, 3, 1).reshape(B, H * W, P)  # [B, H*W, P]
        segments_flat = torch.from_numpy(
            segments).reshape(-1).long().to(device)  # [H*W]

        # Aggregate superpixel features using segment mean
        superpixel_features = torch.zeros(B, num_superpixels, P, device=device)
        for seg_id in range(num_superpixels):
            mask = (segments_flat == seg_id)  # [H*W]
            if mask.any():
                # Correct indexing: fused_flat is [B, H*W, P], mask is [H*W]
                # [B, num_pixels_in_seg, P]
                masked_features = fused_flat[:, mask, :]
                superpixel_features[:, seg_id] = masked_features.mean(
                    dim=1)  # [B, P]

        # Apply attention smoothing
        if attention_matrix is None:
            attention_matrix = torch.eye(num_superpixels, device=device)

        attention_weights = F.softmax(attention_matrix, dim=-1)  # [K, K]
        smoothed_features = torch.bmm(
            attention_weights.unsqueeze(0).expand(B, -1, -1),
            superpixel_features
        )  # [B, K, P]

        # Map back to pixels
        # [B, H*W, P]
        smoothed_abundances_flat = smoothed_features[:, segments_flat]
        smoothed_abundances = smoothed_abundances_flat.reshape(
            B, H, W, P).permute(0, 3, 1, 2)  # [B, P, H, W]

        # Alpha blending
        if alpha is None:
            alpha_val = 0.5
        else:
            alpha_val = torch.clamp(alpha, 0.0, 1.0).item()

        corrected_abundances = alpha_val * smoothed_abundances + \
            (1 - alpha_val) * fused_abundances

        return corrected_abundances

# ==========================================
# 6. Main Model Wrapper
# ==========================================


class MSGACD_Unmixer(nn.Module):
    def __init__(self, num_bands, num_endmembers, H, W, init_M=None):
        super().__init__()
        self.diff_slic = DiffSLIC(num_bands, num_superpixels=120)

        self.gat_encoder = GATEncoder(num_bands, num_endmembers)
        self.cnn_encoder = CNNEncoder(num_bands, num_endmembers)

        self.fusion = CrossAttentionFusion(num_endmembers)
        self.pcr = InterSuperpixelPCR()
        self.decoder = ACDEDecoder(num_endmembers, num_bands, init_M=init_M)

        self.alpha_pcr = nn.Parameter(torch.tensor(0.3))

        self.H = H
        self.W = W
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.warmup_enabled = False

    def build_knn_graph(self, features, k=6):
        """Builds adjacency matrix with SAM (spectral angle) similarity [B, K, C]"""
        B, K, C = features.shape
        adj = torch.zeros(B, K, K, device=features.device)

        features_norm = F.normalize(features, p=2, dim=-1)
        sam_sim = torch.bmm(features_norm, features_norm.transpose(1, 2))
        sam_sim = torch.relu(sam_sim)

        d_sam = 1 - sam_sim
        _, idx = torch.topk(-sam_sim, k + 1, dim=-1)

        for b in range(B):
            for i in range(K):
                neighbors = idx[b, i, 1:]
                adj[b, i, neighbors] = sam_sim[b, i, neighbors]
                adj[b, neighbors, i] = sam_sim[b, neighbors, i]

        return adj

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. Superpixel Branch
        # Q: [B, N, K], Super_feats: [B, K, C]
        Q, super_feats = self.diff_slic(x)

        # Build graph dynamically
        if PYG_AVAILABLE:
            # Conversion to edge_index for PyG usually happens here
            # For simplicity in this demo, we use the custom/matrix flow or a placeholder
            # Real PyG impl would use knn_graph from torch_cluster
            adj = self.build_knn_graph(super_feats)
            edge_index = adj[0].nonzero().t()  # Simple conversion for batch 1
            global_abund = self.gat_encoder(super_feats[0], edge_index)
            global_abund = global_abund.unsqueeze(0)  # [1, K, P]
        else:
            adj = self.build_knn_graph(super_feats)
            global_abund = self.gat_encoder(super_feats, adj)  # [B, K, P]

        # Project global abundances back to pixel space
        # Normalize GAT outputs first to ensure they're in valid range
        global_abund = F.softmax(global_abund, dim=-1)

        # A_global = Q * A_nodes
        # [B, N, K] x [B, K, P] -> [B, N, P]
        global_abund_pixel = torch.bmm(Q, global_abund)
        global_abund_pixel = global_abund_pixel.permute(
            0, 2, 1).view(B, -1, H, W)

        # Ensure global abundances are normalized
        global_abund_pixel = F.softmax(global_abund_pixel, dim=1)

        # 2. Local CNN Branch
        local_abund = self.cnn_encoder(x)  # [B, P, H, W]
        local_abund = F.softmax(local_abund, dim=1)

        # 3. Fusion
        fused_abund = self.fusion(global_abund_pixel, local_abund)

        # 4. Inter-Superpixel PCR smoothing (disabled for now)
        smoothed_abund = fused_abund

        # 5. Reconstruction
        reconstruction, endmembers = self.decoder(smoothed_abund, x)

        return reconstruction, smoothed_abund, endmembers, global_abund, adj

# ==========================================
# 7. Loss Functions & Training
# ==========================================


def sad_loss(y_true, y_pred):
    h_true, w_true = y_true.shape[2], y_true.shape[3]
    h_pred, w_pred = y_pred.shape[2], y_pred.shape[3]
    h_min, w_min = min(h_true, h_pred), min(w_true, w_pred)
    y_true = y_true[:, :, :h_min, :w_min]
    y_pred = y_pred[:, :, :h_min, :w_min]

    y_true_norm = F.normalize(y_true, p=2, dim=1)
    y_pred_norm = F.normalize(y_pred, p=2, dim=1)
    dot = torch.sum(y_true_norm * y_pred_norm, dim=1)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(dot)
    return torch.mean(angle)


def soft_clustering_loss(abundances, endmembers):
    """
    ACDE Prior: Encourages pixels to cluster around endmembers.
    Simplified version of DEC loss.
    """
    # Just entropy minimization for sharpening
    return -torch.mean(torch.sum(abundances * torch.log(abundances + 1e-8), dim=1))


def dirichlet_loss(abundances, alpha=1.0):
    """
    Dirichlet prior: Encourages sparsity and pure pixels.
    """
    # Dirichlet KL divergence
    dirichlet_prior = torch.ones_like(abundances) * alpha
    kl = torch.sum(abundances * (torch.log(abundances + 1e-8) -
                                 torch.log(dirichlet_prior + 1e-8)), dim=1)
    return torch.mean(kl)


def spatial_tv_loss(abundances):
    """
    Total Variation loss for spatial smoothness of abundances.
    abundances: [B, P, H, W]
    """
    B, P, H, W = abundances.shape

    # Horizontal gradients
    diff_h = abundances[:, :, :, 1:] - abundances[:, :, :, :-1]

    # Vertical gradients
    diff_v = abundances[:, :, 1:, :] - abundances[:, :, :-1, :]

    tv_loss = torch.mean(torch.abs(diff_h)) + torch.mean(torch.abs(diff_v))
    return tv_loss


def reweighted_tv_loss(abundances, image, alpha=1.0):
    """
    Adaptive Abundance Smoothing (AAS) with Reweighted TV.
    Smooths abundances strongly in homogeneous regions but preserves edges.

    abundances: [B, P, H, W]
    image: [B, C, H, W] - original hyperspectral image
    alpha: edge sensitivity parameter
    """
    B, P, H, W = abundances.shape

    # Compute edge strength from original image
    # Use spectral gradient magnitude
    img_grad_h = image[:, :, :, 1:] - image[:, :, :, :-1]  # [B, C, H, W-1]
    img_grad_v = image[:, :, 1:, :] - image[:, :, :-1, :]  # [B, C, H-1, W]

    edge_strength_h = torch.norm(img_grad_h, p=2, dim=1)  # [B, H, W-1]
    edge_strength_v = torch.norm(img_grad_v, p=2, dim=1)  # [B, H-1, W]

    # Compute weights (inverse of edge strength)
    w_h = torch.exp(-alpha * edge_strength_h)  # [B, H, W-1]
    w_v = torch.exp(-alpha * edge_strength_v)  # [B, H-1, W]

    # Abundance gradients
    abund_grad_h = abundances[:, :, :, 1:] - \
        abundances[:, :, :, :-1]  # [B, P, H, W-1]
    abund_grad_v = abundances[:, :, 1:, :] - \
        abundances[:, :, :-1, :]  # [B, P, H-1, W]

    # Reweighted TV: weighted L1 norm of gradients
    rtv_h = torch.abs(abund_grad_h) * w_h.unsqueeze(1)  # [B, P, H, W-1]
    rtv_v = torch.abs(abund_grad_v) * w_v.unsqueeze(1)  # [B, P, H-1, W]

    rtv_loss = torch.mean(rtv_h) + torch.mean(rtv_v)
    return rtv_loss


def endmember_orthogonality_loss(endmembers):
    """
    Encourages endmembers to be orthogonal (reduces mixing confusion).
    endmembers: [C, P]
    """
    M_norm = F.normalize(endmembers, p=2, dim=0)  # Normalize each endmember
    gram = torch.matmul(M_norm.t(), M_norm)  # [P, P]
    # Penalize off-diagonal elements (should be close to 0)
    mask = ~torch.eye(gram.shape[0], dtype=torch.bool, device=gram.device)
    return torch.mean(gram[mask]**2)


def train_unmixer(data_path=None, endmember_path=None):
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Ground Truth Data if available
    M_true = None
    A_true_reshaped = None

    # Try to load samson data
    samson_path = 'data/samson.mat'
    end3_path = 'data/end3.mat' if endmember_path is None else endmember_path

    if os.path.exists(samson_path) and os.path.exists(end3_path):
        print("Loading ground truth data...")
        samson_data = scipy.io.loadmat(samson_path)
        endmember_data = scipy.io.loadmat(end3_path)

        # Extract shape information
        nRow = int(samson_data['nRow'][0][0])
        nCol = int(samson_data['nCol'][0][0])
        nBand = int(samson_data['nBand'][0][0])

        # Extract hyperspectral data
        V = samson_data['V']  # [nBand, nRow*nCol]
        img = V.T.reshape(nRow, nCol, nBand)  # [nRow, nCol, nBand]

        # Extract ground truth endmembers and abundances
        M_true = endmember_data['M']  # [nBand, num_endmembers]
        A_true = endmember_data['A']  # [num_endmembers, nRow*nCol]

        num_endmembers = M_true.shape[1]

        # Reshape ground truth abundances for evaluation
        A_true_reshaped = np.zeros((nRow, nCol, num_endmembers))
        for i in range(num_endmembers):
            A_true_reshaped[:, :, i] = A_true[i, :].reshape(nRow, nCol)

        # Normalize
        img = (img - np.min(img)) / (np.max(img) - np.min(img))

        print(
            f"Loaded Samson data: {img.shape}, Endmembers: {M_true.shape}, Abundances: {A_true.shape}")
    elif data_path and os.path.exists(data_path):
        mat = scipy.io.loadmat(data_path)
        img = mat['paviaU'] if 'paviaU' in mat else mat['V']
        if len(img.shape) == 2:
            # Reshape if needed
            nRow = int(np.sqrt(img.shape[1]))
            nCol = nRow
            nBand = img.shape[0]
            img = img.T.reshape(nRow, nCol, nBand)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        num_endmembers = 4
    else:
        print("Data file not found. Generating dummy hyperspectral cube...")
        H, W, Bands = 64, 64, 100
        img = np.random.rand(H, W, Bands).astype(np.float32)
        img[:32, :32, :] += 0.5
        num_endmembers = 4

    # Prepare Tensor
    if len(img.shape) == 3:
        img_tensor = torch.from_numpy(img).permute(
            2, 0, 1).unsqueeze(0).float().to(device)
    else:
        raise ValueError(f"Unexpected image shape: {img.shape}")

    num_bands = img_tensor.shape[1]
    H, W = img_tensor.shape[2], img_tensor.shape[3]

    if M_true is not None:
        num_endmembers = M_true.shape[1]

    # VCA initialization for endmembers
    init_M = None
    if os.path.exists(samson_path):
        try:
            img_flat = img.reshape(-1, num_bands)
            init_M = vca_initialization(img_flat, num_endmembers)
            print("VCA initialization successful")
        except Exception as e:
            print(f"VCA initialization failed: {e}, using random init")

    # Initialize Model
    model = MSGACD_Unmixer(num_bands, num_endmembers, H,
                           W, init_M=init_M).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20)

    epochs = 250
    losses = []

    print("Starting training...")
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        if epoch < 60:
            for p in model.gat_encoder.parameters():
                p.requires_grad = False
        else:
            for p in model.gat_encoder.parameters():
                p.requires_grad = True

        model.warmup_enabled = False

        recon, abundances, endmembers, global_abund, adj = model(img_tensor)

        # Use normalized abundances for losses (fused abundances are already softmax'd)
        A = abundances  # Already normalized from fusion

        h_in, w_in = img_tensor.shape[2], img_tensor.shape[3]
        h_recon, w_recon = recon.shape[2], recon.shape[3]
        h_min, w_min = min(h_in, h_recon), min(w_in, w_recon)
        img_crop = img_tensor[:, :, :h_min, :w_min]
        recon_crop = recon[:, :, :h_min, :w_min]

        l_sad = sad_loss(img_crop, recon_crop)
        l_mse = F.mse_loss(img_crop, recon_crop)

        l_sparse = torch.mean(torch.sqrt(
            A + 1e-8)) if epoch >= 150 else torch.tensor(0.0, device=device)
        l_cluster = soft_clustering_loss(
            A, endmembers) if epoch >= 150 else torch.tensor(0.0, device=device)

        diff = endmembers[1:, :] - endmembers[:-1, :]
        l_smooth = torch.mean(
            diff**2) if epoch >= 50 else torch.tensor(0.0, device=device)

        # Graph Laplacian regularization for smoothness (FIXED: use trace)
        if epoch >= 200:
            B, K, P = global_abund.shape
            adj_float = adj.float()
            deg = torch.sum(adj_float, dim=-1, keepdim=True)  # [B, K, 1]
            lap = torch.diag_embed(deg.squeeze(-1)) - adj_float  # [B, K, K]

            # Smoothness: trace(A^T * L * A) - correct formulation
            lap_A = torch.bmm(lap, global_abund)  # [B, K, P]
            smooth_graph = torch.bmm(
                global_abund.transpose(1, 2), lap_A)  # [B, P, P]
            l_graph = torch.mean(torch.diagonal(
                smooth_graph, dim1=1, dim2=2))  # [B, P] -> scalar
        else:
            l_graph = torch.tensor(0.0, device=device)

        l_endmember_sad = torch.tensor(0.0, device=device)
        if M_true is not None and epoch >= 50:
            for i in range(num_endmembers):
                M_est_norm = F.normalize(endmembers[:, i], p=2, dim=0)
                M_true_tensor = torch.from_numpy(
                    M_true[:, i]).float().to(device)
                M_true_norm = F.normalize(M_true_tensor, p=2, dim=0)
                dot = torch.sum(M_est_norm * M_true_norm)
                dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
                angle = torch.acos(dot)
                l_endmember_sad += angle
            l_endmember_sad = l_endmember_sad / num_endmembers

        if epoch < 200:
            w_sad = 3.0
            w_mse = 0.5
            w_sparse = 0.0
            w_cluster = 0.0
            w_smooth = 0.0
            w_graph = 0.0
            w_dirichlet = 0.0
            w_tv = 0.0
            w_ortho = 0.0
            w_endmember_sad = 0.0
            l_dirichlet = torch.tensor(0.0, device=device)
            l_tv_spatial = torch.tensor(0.0, device=device)
            l_ortho = torch.tensor(0.0, device=device)
            l_graph = torch.tensor(0.0, device=device)
        else:
            w_sad = 3.0
            w_mse = 0.5
            w_sparse = 0.0
            w_cluster = 0.0
            w_smooth = 0.0
            w_graph = 0.05
            w_dirichlet = 0.0
            w_tv = 0.02
            w_ortho = 0.0
            w_endmember_sad = 0.0
            l_dirichlet = torch.tensor(0.0, device=device)
            l_tv_spatial = reweighted_tv_loss(A, img_crop, alpha=1.0)
            l_ortho = torch.tensor(0.0, device=device)

        total_loss = (
            w_sad * l_sad +
            w_mse * l_mse +
            w_sparse * l_sparse +
            w_cluster * l_cluster +
            w_smooth * l_smooth +
            w_graph * l_graph +
            w_dirichlet * l_dirichlet +
            w_tv * l_tv_spatial +
            w_ortho * l_ortho +
            w_endmember_sad * l_endmember_sad
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step(total_loss)

        losses.append({
            'total': total_loss.item(),
            'sad': l_sad.item(),
            'mse': l_mse.item(),
            'lq': l_sparse.item(),
            'smooth': l_smooth.item(),
            'graph': l_graph.item(),
            'dirichlet': l_dirichlet.item(),
            'tv': l_tv_spatial.item(),
            'ortho': l_ortho.item()
        })

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Total: {total_loss.item():.4f} | "
                  f"SAD: {l_sad.item():.4f} | MSE: {l_mse.item():.4f} | "
                  f"TV: {l_tv_spatial.item():.4f} | Graph: {l_graph.item():.4f}")

    print("Training completed!")

    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        recon, abundances, extracted_endmembers, global_abund, _ = model(
            img_tensor)

        abund_maps = abundances[0].cpu().numpy()  # [P, H, W]
        abund_maps = abund_maps.transpose(1, 2, 0)  # [H, W, P]

        endmembers_np = extracted_endmembers.cpu().numpy()  # [C, P]
        endmembers_np = endmembers_np.T  # [P, C] for plotting

    # 1. Training Losses Plot
    plt.figure(figsize=(10, 6))
    plt.plot([l['total'] for l in losses], label='Total Loss')
    plt.plot([l['sad'] for l in losses], label='SAD Loss')
    plt.plot([l['mse'] for l in losses], label='MSE Loss')
    plt.plot([l['lq'] for l in losses], label='Lq Norm')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('results/training_losses.png')
    plt.close()

    # 2. Abundance Maps Comparison
    if A_true_reshaped is not None:
        # Estimated vs Ground Truth
        plt.figure(figsize=(18, 12))
        for i in range(num_endmembers):
            # Estimated abundance maps
            plt.subplot(2, num_endmembers, i + 1)
            plt.imshow(abund_maps[:, :, i], cmap='jet')
            plt.colorbar()
            plt.title(f'Estimated Abundance {i+1}')
            plt.axis('off')

            # Ground truth
            plt.subplot(2, num_endmembers, i + 1 + num_endmembers)
            plt.imshow(A_true_reshaped[:, :, i], cmap='jet')
            plt.colorbar()
            plt.title(f'Ground Truth Abundance {i+1}')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('results/abundance_comparison.png')
        plt.close()
    else:
        # Just show estimated abundances if no ground truth
        plt.figure(figsize=(15, 5))
        for i in range(num_endmembers):
            plt.subplot(1, num_endmembers, i + 1)
            plt.imshow(abund_maps[:, :, i], cmap='jet')
            plt.colorbar()
            plt.title(f'Estimated Abundance {i+1}')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('results/abundance_comparison.png')
        plt.close()

    # 3. Endmembers Comparison
    plt.figure(figsize=(10, 6))
    for i in range(num_endmembers):
        plt.plot(range(num_bands), endmembers_np[i, :],
                 label=f'Estimated Endmember {i+1}')
        if M_true is not None:
            plt.plot(range(num_bands), M_true[:, i], '--',
                     label=f'True Endmember {i+1}')
    plt.title('Endmembers Comparison')
    plt.xlabel('Band')
    plt.ylabel('Reflectance')
    plt.legend()
    plt.savefig('results/endmembers_comparison.png')
    plt.close()

    print("Results saved to results/ directory:")
    print("  - results/training_losses.png")
    print("  - results/abundance_comparison.png")
    print("  - results/endmembers_comparison.png")


if __name__ == "__main__":
    # Try to load samson data, or use custom path
    train_unmixer(data_path=None, endmember_path=None)
