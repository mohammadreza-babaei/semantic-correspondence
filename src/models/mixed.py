
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedModelAdapter:
    def __init__(self, adapter1, adapter2, mix_weights=None):
        """
        Adapter that runs two models in parallel and concatenates their features.
        
        Args:
            adapter1: First model adapter (output size determines primary grid)
            adapter2: Second model adapter
            mix_weights: Optional list/tuple of weights [w1, w2] for the features
        """
        self.adapter1 = adapter1
        self.adapter2 = adapter2
        self.mix_weights = mix_weights
        self.device = adapter1.device
        self.model_name = f"mixed_{adapter1.model_name}_{adapter2.model_name}"
        
        # Use adapter1 as primary for sizing
        self.standard_size = adapter1.standard_size 
        
        # Dummy values for cache naming compatibility
        self.num_frozen_blocks = adapter1.num_frozen_blocks 
        self.num_unfrozen_blocks = adapter1.num_unfrozen_blocks
        
        # Expose underlying models in a specific way if needed by optimization
        # (Though evaluating mixed models usually assumes they are already trained)
        self.model = nn.ModuleDict({
            'model1': adapter1.model if hasattr(adapter1, 'model') else nn.Identity(),
            'model2': adapter2.model if hasattr(adapter2, 'model') else nn.Identity()
        })
        
        self.trainable_params = [] # No training logic for mixed adapter yet

    def preprocess_image(self, image_path, target_size=None):
        return self.adapter1.preprocess_image(image_path, target_size)

    def preprocess_image_pil(self, img, target_size=None):
        return self.adapter1.preprocess_image_pil(img, target_size)

    def extract_intermediate_features(self, image_tensor):
        """
        Extract features for both models.
        Handles resizing the input tensor if the second model expects a different size.
        """
        # 1. Adapter 1
        feat1 = self.adapter1.extract_intermediate_features(image_tensor)
        
        # 2. Adapter 2
        # Check standard sizes compatibility
        feat2 = None
        if self.adapter1.standard_size != self.adapter2.standard_size:
             # Resize image_tensor to adapter2's expected size
            target_size = self.adapter2.standard_size
            
            # image_tensor is (B, C, H, W). We use bilinear interpolation.
            img_resized = F.interpolate(
                image_tensor, 
                size=(target_size, target_size), 
                mode='bilinear', 
                align_corners=False
            )
            feat2 = self.adapter2.extract_intermediate_features(img_resized)
        else:
            feat2 = self.adapter2.extract_intermediate_features(image_tensor)
            
        return {'f1': feat1, 'f2': feat2}

    def forward_unfrozen_blocks(self, intermediate_features):
        """
        Run both models forward and mix (concatenate) the results.
        Propagates gradients if underlying models allow it.
        """
        f1 = intermediate_features['f1']
        f2 = intermediate_features['f2']
        
        out1 = self.adapter1.forward_unfrozen_blocks(f1)
        out2 = self.adapter2.forward_unfrozen_blocks(f2)
        
        # out1: (B, C1, H1, W1)
        # out2: (B, C2, H2, W2)
        
        # Interpolate out2 to match out1's spatial dims if necessary
        if out1.shape[2:] != out2.shape[2:]:
            out2 = F.interpolate(
                out2,
                size=(out1.shape[2], out1.shape[3]),
                mode='bilinear',
                align_corners=False
            )
        
        # Normalize features before combining to ensure equal contribution
        out1 = F.normalize(out1, p=2, dim=1)
        out2 = F.normalize(out2, p=2, dim=1)
        
        # Apply mixing weights if configured
        if self.mix_weights is not None:
            out1 = out1 * self.mix_weights[0]
            out2 = out2 * self.mix_weights[1]
            
        # Concatenate along channel dimension
        return torch.cat([out1, out2], dim=1)
        
    def get_model_state(self):
        return {
            "model1_name": self.adapter1.model_name,
            "model2_name": self.adapter2.model_name,
        }
    
    def load_model_state(self, state):
        # We assume wrappers are loaded individually before creation of MixedModelAdapter
        pass
