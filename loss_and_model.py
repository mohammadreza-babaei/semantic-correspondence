import torch
import torch.nn as nn
import torch.nn.functional as F

# Tenta di importare il tuo estrattore. 
# Assicurati che il file 'dinov2_features.py' sia nella stessa directory.
try:
    from dinov2_features import DinoV2FeatureExtractor
except ImportError:
    raise ImportError("Non riesco a trovare 'dinov2_features.py'. Assicurati che sia nella stessa cartella.")

class SoftArgmax2D(nn.Module):
    """
    Calcola le coordinate (x, y) differenziabili dal centro di massa della heatmap.
    Essenziale per la componente L_sparse della loss.
    """
    def __init__(self, temperature=100.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, heatmaps):
        """
        Args:
            heatmaps: (Batch, Num_Keypoints, H, W)
        Returns:
            coords: (Batch, Num_Keypoints, 2) -> [x, y] in coordinate pixel
        """
        B, N, H, W = heatmaps.shape
        
        # 1. Appiattisci spazialmente (B, N, H*W)
        heatmaps_flat = heatmaps.view(B, N, -1)
        
        # 2. Softmax spaziale (la temperatura controlla quanto è "appuntita" la distribuzione)
        weights = F.softmax(heatmaps_flat * self.temperature, dim=2)
        weights = weights.view(B, N, H, W)
        
        # 3. Griglie di coordinate
        # Nota: indexing='xy' produce griglie dove x varia lungo le colonne e y lungo le righe
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        
        grid_x = grid_x.to(heatmaps.device).float()
        grid_y = grid_y.to(heatmaps.device).float()
        
        # 4. Somma pesata (Valore Atteso)
        expected_x = torch.sum(weights * grid_x, dim=(2, 3))
        expected_y = torch.sum(weights * grid_y, dim=(2, 3))
        
        return torch.stack([expected_x, expected_y], dim=-1)

class SparseDenseLoss(nn.Module):
    """
    Implementazione della loss ibrida dal paper 'Telling Left from Right'.
    Loss = L_sparse + alpha * L_dense
    """
    def __init__(self, alpha=0.1, sigma=5.0, image_size=(224, 224)):
        super().__init__()
        self.alpha = alpha       # Peso per la dense loss (default 0.1 come nel paper)
        self.sigma = sigma       # Sigma per le Gaussiane target
        self.soft_argmax = SoftArgmax2D(temperature=100.0)
        self.image_size = image_size

    def generate_gaussian_targets(self, coords, H, W, device):
        """Crea le heatmap target M* con gaussiane centrate sulle coordinate GT."""
        B, N, _ = coords.shape
        target_maps = torch.zeros(B, N, H, W, device=device)
        
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        grid_x = grid_x.to(device).float()
        grid_y = grid_y.to(device).float()
        
        for b in range(B):
            for n in range(N):
                # Se le coordinate sono negative o fuori dall'immagine (non visibili),
                # la gaussiana sarà zero o fuori campo (gestito implicitamente o tramite visibility mask dopo)
                cx, cy = coords[b, n]
                
                # Formula Gaussiana non normalizzata
                squared_dist = (grid_x - cx)**2 + (grid_y - cy)**2
                g = torch.exp(-squared_dist / (2 * self.sigma**2))
                target_maps[b, n] = g
                
        return target_maps

    def forward(self, pred_heatmaps, gt_coords, visibility):
        """
        Args:
            pred_heatmaps: (B, N, H, W) Output del modello
            gt_coords: (B, N, 2) Coordinate vere [x, y]
            visibility: (B, N) Flag 1.0 se il punto è visibile, 0.0 altrimenti
        """
        # --- 1. DENSE LOSS (MSE pixel-wise) ---
        # Generiamo le target maps M* dinamicamente
        B, N, H, W = pred_heatmaps.shape
        target_heatmaps = self.generate_gaussian_targets(gt_coords, H, W, pred_heatmaps.device)
        
        # MSE Loss: || M - M* ||^2
        # Usiamo reduction='none' per applicare la visibilità manualmente se necessario
        mse_loss = F.mse_loss(pred_heatmaps, target_heatmaps, reduction='none')
        
        # Media su H e W per ottenere l'errore per keypoint
        mse_per_keypoint = mse_loss.mean(dim=(2, 3)) # (B, N)
        
        # La dense loss nel paper è mediata su tutti i punti. 
        # Moltiplichiamo per visibility per ignorare punti occulti nel training denso (best practice)
        dense_loss = (mse_per_keypoint * visibility).sum() / (visibility.sum() + 1e-8)

        # --- 2. SPARSE LOSS (L2 coordinate) ---
        # Estraiamo coordinate predette con Soft-Argmax
        pred_coords = self.soft_argmax(pred_heatmaps)
        
        # Distanza Euclidea quadrata: (x - x_hat)^2 + (y - y_hat)^2
        # gt_coords deve essere float
        sq_distance = (gt_coords - pred_coords).pow(2).sum(dim=-1) # (B, N)
        
        # Somma pesata dalla visibilità normalizzata
        # L_sparse = (1 / sum(v)) * sum(v * dist)
        sparse_loss = (sq_distance * visibility).sum() / (visibility.sum() + 1e-8)
        
        # --- LOSS TOTALE ---
        total_loss = sparse_loss + self.alpha * dense_loss
        
        return total_loss, sparse_loss, dense_loss

class DinoV2KeypointDetector(nn.Module):
    def __init__(self, model_name='dinov2_vits14', num_keypoints=10, freeze_backbone=True):
        super().__init__()
        # 1. Backbone: Usa il tuo estrattore esistente
        self.feature_extractor = DinoV2FeatureExtractor(model_name)
        
        # Congela il backbone se richiesto
        if freeze_backbone:
            for param in self.feature_extractor.model.parameters():
                param.requires_grad = False
            self.feature_extractor.model.eval() # Imposta in eval mode (utile per Dropout/BatchNorm)

        # Recupera dimensione embedding (es. 384)
        embed_dim = self.feature_extractor.model.embed_dim
        
        # 2. Prediction Head (Conv 1x1)
        # Trasforma i 384 canali in N heatmap (una per keypoint)
        self.head = nn.Conv2d(embed_dim, num_keypoints, kernel_size=1)
        
        # Inizializzazione pesi della testa (opzionale ma consigliata)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        x: Immagini (B, 3, H, W)
        Returns: Heatmaps ad alta risoluzione (B, N, H, W)
        """
        input_h, input_w = x.shape[2], x.shape[3]
        
        # Feature Extraction -> Output atteso: (B, H_grid, W_grid, D)
        features = self.feature_extractor(x)
        
        # Permuta per Conv2d: (B, D, H_grid, W_grid)
        features = features.permute(0, 3, 1, 2)
        
        # Genera heatmap a bassa risoluzione (grid size di Dino)
        heatmaps_low_res = self.head(features)
        
        # Upsampling bilineare per matchare la risoluzione originale
        # Questo permette alla loss densa e al soft-argmax di lavorare con precisione pixel
        heatmaps_high_res = F.interpolate(
            heatmaps_low_res, 
            size=(input_h, input_w), 
            mode='bilinear', 
            align_corners=False
        )
        
        return heatmaps_high_res

# --- Test use
if __name__ == "__main__":
    # Configurazione Dummy
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_KEYPOINTS = 5
    IMG_SIZE = 224 # DinoV2 preferisce multipli di 14 (es. 224, 518)
    
    print(f"Inizializzazione modello su {DEVICE}...")
    
    # 1. Istanzia il Modello
    # model_name deve corrispondere a quelli gestiti dal tuo dinov2_features.py
    model = DinoV2KeypointDetector(model_name='dinov2_vits14', num_keypoints=NUM_KEYPOINTS).to(DEVICE)
    
    # 2. Istanzia la Loss
    criterion = SparseDenseLoss(alpha=0.1, sigma=3.0).to(DEVICE)
    
    # 3. Ottimizzatore (solo per la testa se il backbone è congelato)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3)
    
    # 4. Generazione Dati Finti (Dummy Data)
    print("Generazione batch di prova...")
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    
    # Coordinate GT (x, y) casuali tra 0 e IMG_SIZE
    dummy_coords = torch.rand(batch_size, NUM_KEYPOINTS, 2).to(DEVICE) * IMG_SIZE
    
    # Visibility: facciamo finta che il primo punto della prima immagine sia nascosto (0)
    dummy_vis = torch.ones(batch_size, NUM_KEYPOINTS).to(DEVICE)
    dummy_vis[0, 0] = 0.0 
    
    #Training Loop Simulato
    print("Esecuzione Forward e Backward pass...")
    
    # Forward
    pred_heatmaps = model(dummy_images)
    print(f"Output Heatmaps shape: {pred_heatmaps.shape}")
    
    # Calcolo Loss
    loss, l_sparse, l_dense = criterion(pred_heatmaps, dummy_coords, dummy_vis)
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("-" * 30)
    print(f"Sparse Loss: {l_sparse.item():.4f}")
    print(f"Dense Loss:  {l_dense.item():.4f}")
    print(f"Total Loss:  {loss.item():.4f}")
    print("-" * 30)
    print("Test Completed")