import torch
import logging
from torch.autograd import Function
from typing import List, Dict, Any, Tuple
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from ..recstore.KVClient import get_kv_client, RecStoreClient

# Configure logging for better diagnostics
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class _RecStoreEBCFunction(Function):
    """
    FX-traceable autograd Function to bridge PyTorch's autograd engine with
    the RecStore backend.
    """

    @staticmethod
    def forward(
        ctx,
        # Non-tensor arguments are passed first
        module: "RecStoreEmbeddingBagCollection",
        feature_keys: List[str],
        # Tensor arguments (which become Proxies during tracing) are passed next
        features_values: torch.Tensor,
        features_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Performs the forward pass by pulling embedding vectors from RecStore.
        This implementation is designed to be torch.fx.traceable.
        """
        # Check if we're in FX tracing mode
        is_tracing = hasattr(features_values, 'node') and hasattr(features_values.node, 'op')
        
        if is_tracing:
            # During FX tracing, we need to return a placeholder tensor
            # that represents the pooled embeddings for each feature
            # The shape should be (batch_size, num_features, embedding_dim)
            embedding_dim = module._embedding_dims[feature_keys[0]] if feature_keys else 64
            num_features = len(feature_keys)
            # During tracing, we can't access device attribute, so use CPU
            return torch.zeros(1, num_features, embedding_dim, device='cpu')
        
        # Normal execution path
        if not isinstance(features_lengths, torch.Tensor):
            lengths_tensor = torch.tensor(features_lengths, dtype=torch.int32)
        else:
            lengths_tensor = features_lengths.to(dtype=torch.int32, device="cpu")
        
        features = KeyedJaggedTensor(
            keys=feature_keys,
            values=features_values,
            lengths=lengths_tensor,
        )
        
        ctx.save_for_backward(features_values, features_lengths)
        ctx.module = module
        ctx.feature_keys = feature_keys

        # Get the batch size from the first feature's length
        batch_size = len(features.lengths()) // len(feature_keys)
        embedding_dim = module._embedding_dims[feature_keys[0]]
        
        # Add debug logging
        # logging.info(f"Debug: batch_size={batch_size}, num_features={len(feature_keys)}, embedding_dim={embedding_dim}")
        # logging.info(f"Debug: lengths shape={len(features.lengths())}, values shape={features.values().shape}")
        
        # Initialize output tensor: (batch_size, num_features, embedding_dim)
        output = torch.zeros(batch_size, len(feature_keys), embedding_dim, 
                           device=features_values.device, dtype=torch.float32)
        
        lengths = features.lengths()
        values = features.values()
        
        # Process each feature
        for feature_idx in range(len(feature_keys)):
            config_name = module._config_names[feature_keys[feature_idx]]
            
            # Process each sample in the batch
            for sample_idx in range(batch_size):
                # Get the length for this sample and this feature
                length_idx = sample_idx * len(feature_keys) + feature_idx
                l = lengths[length_idx]
                
                if l > 0:
                    # Calculate the start index for this sample and feature
                    start_idx = 0
                    for i in range(length_idx):
                        start_idx += lengths[i]
                    
                    # Get the IDs for this sample and feature
                    ids_to_pull = values[start_idx:start_idx + l]
                    
                    # Pull embeddings from RecStore
                    embeddings = module.kv_client.pull(name=config_name, ids=ids_to_pull)
                    
                    # Average the embeddings for this sample and feature
                    pooled_emb = embeddings.mean(dim=0)
                    output[sample_idx, feature_idx] = pooled_emb
                else:
                    # If no embeddings, keep as zero (already initialized)
                    pass
        
        return output

    @staticmethod
    def backward(
        ctx, grad_output_values: torch.Tensor
    ) -> Tuple[None, None, None, None]:
        """
        Performs the backward pass by pushing gradients to the RecStore backend.
        This implementation is also FX-traceable.
        """
        features_values, features_lengths = ctx.saved_tensors
        module: "RecStoreEmbeddingBagCollection" = ctx.module
        feature_keys: List[str] = ctx.feature_keys

        # Check if we're in FX tracing mode
        is_tracing = hasattr(features_values, 'node') and hasattr(features_values.node, 'op')
        
        if is_tracing:
            # During FX tracing, we don't need to do actual gradient updates
            # Just return None gradients for all inputs
            return None, None, None, None

        # Normal execution path
        if not isinstance(features_lengths, torch.Tensor):
            lengths_tensor = torch.tensor(features_lengths, dtype=torch.int32)
        else:
            lengths_tensor = features_lengths.to(dtype=torch.int32, device="cpu")
        features = KeyedJaggedTensor(
            keys=feature_keys,
            values=features_values,
            lengths=lengths_tensor,
        )
        
        # grad_output_values shape is (batch_size * num_features, embedding_dim)
        # We need to reshape it back to (batch_size, num_features, embedding_dim)
        batch_size = len(features.lengths()) // len(feature_keys)
        num_features = len(feature_keys)
        embedding_dim = grad_output_values.shape[-1]
        
        grad_output_reshaped = grad_output_values.reshape(batch_size, num_features, embedding_dim)
        
        # Add debug logging
        # logging.info(f"Debug backward: batch_size={batch_size}, num_features={num_features}, embedding_dim={embedding_dim}")
        # logging.info(f"Debug backward: grad_output_values shape={grad_output_values.shape}, grad_output_reshaped shape={grad_output_reshaped.shape}")
        
        lengths = features.lengths()
        values = features.values()
        
        # Process each feature
        for feature_idx in range(len(feature_keys)):
            config_name = module._config_names[feature_keys[feature_idx]]
            
            # Process each sample in the batch
            for sample_idx in range(batch_size):
                # Get the length for this sample and this feature
                length_idx = sample_idx * len(feature_keys) + feature_idx
                l = lengths[length_idx]
                
                if l > 0:
                    # Calculate the start index for this sample and feature
                    start_idx = 0
                    for i in range(length_idx):
                        start_idx += lengths[i]
                    
                    # Get the IDs for this sample and feature
                    ids_to_update = values[start_idx:start_idx + l]
                    
                    # Get the gradient for this sample and feature
                    grad = grad_output_reshaped[sample_idx, feature_idx]
                    
                    # Expand gradient to match the number of embeddings
                    grads = grad.unsqueeze(0).expand(l, -1)
                    
                    # Update embeddings in RecStore
                    module.kv_client.update(name=config_name, ids=ids_to_update, grads=grads.contiguous())

        # Return gradients for inputs to forward: module, feature_keys, features_values, features_lengths
        return None, None, None, None


class RecStoreEmbeddingBagCollection(torch.nn.Module):
    """
    An FX-traceable EmbeddingBagCollection that uses a custom RecStore backend.
    It is designed as a drop-in replacement for torchrec.EmbeddingBagCollection
    within a DLRM model.
    """

    def __init__(self, embedding_bag_configs: List[Dict[str, Any]]):
        super().__init__()

        if not embedding_bag_configs:
            raise ValueError("embedding_bag_configs cannot be empty.")

        # Convert dicts to EmbeddingBagConfig objects
        self._embedding_bag_configs = [
            EmbeddingBagConfig(
                name=c["name"],
                embedding_dim=c["embedding_dim"],
                num_embeddings=c["num_embeddings"],
                feature_names=c.get("feature_names", [c["name"]])
            )
            for c in embedding_bag_configs
        ]
        self.kv_client: RecStoreClient = get_kv_client()
        
        # Store feature names and embedding dimensions in a static, ordered way.
        # This is crucial for FX tracing, as we will iterate over these static lists.
        # We need to use feature_names instead of name to match what SparseArch expects
        self.feature_keys: List[str] = []
        self._embedding_dims: Dict[str, int] = {}
        self._config_names: Dict[str, str] = {}  # Map feature_name to config.name
        for c in self._embedding_bag_configs:
            for feature_name in c.feature_names:
                self.feature_keys.append(feature_name)
                self._embedding_dims[feature_name] = c.embedding_dim
                self._config_names[feature_name] = c.name

        logging.info("Initializing RecStoreEmbeddingBagCollection...")
        for config in self._embedding_bag_configs:
            self._validate_config(config)
            name: str = config.name
            num_embeddings: int = config.num_embeddings
            embedding_dim: int = config.embedding_dim

            logging.info(
                f"  - Ensuring table '{name}' exists in RecStore backend with "
                f"shape ({num_embeddings}, {embedding_dim})."
            )
            self.kv_client.init_data(
                name=name,
                shape=(num_embeddings, embedding_dim),
                dtype=torch.float32,
            )
        logging.info("RecStoreEmbeddingBagCollection initialized successfully.")

    def embedding_bag_configs(self):
        return self._embedding_bag_configs

    def _validate_config(self, config: EmbeddingBagConfig):
        """Helper to validate a single embedding table configuration."""
        required_keys = ["name", "num_embeddings", "embedding_dim"]
        for key in required_keys:
            if not hasattr(config, key):
                raise ValueError(
                    f"Missing required key '{key}' in embedding_bag_configs."
                )

    def forward(self, features: KeyedJaggedTensor) -> KeyedTensor:
        """
        Performs the embedding lookup in a way that is compatible with torch.fx.
        """
        values = features.values().contiguous()
        lengths = features.lengths().contiguous()

        pooled_embs_values = _RecStoreEBCFunction.apply(
            self,
            self.feature_keys,
            values,
            lengths,
        )

        # pooled_embs_values shape is (batch_size, num_features, embedding_dim)
        # We need to reshape it to (batch_size * num_features, embedding_dim) for KeyedTensor
        batch_size, num_features, embedding_dim = pooled_embs_values.shape
        reshaped_values = pooled_embs_values.reshape(-1, embedding_dim)
        
        # Create length_per_key where each feature has batch_size length
        length_per_key = [batch_size] * num_features
        
        return KeyedTensor(
            keys=self.feature_keys,
            values=reshaped_values,
            length_per_key=length_per_key,
            key_dim=0,  # Split along dimension 0
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tables={self.feature_keys})"
