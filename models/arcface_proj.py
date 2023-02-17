from torch import nn
from diffusers.models.attention import BasicTransformerBlock


class ArcFaceProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 512
        self.proj_size = 768
        self.n_dim = 10
        self.mapper = BasicTransformerBlock(dim=self.hidden_size, num_attention_heads=1, attention_head_dim=self.hidden_size, activation_fn="gelu", attention_bias=True)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.proj_size * 10)

    def forward(self, embedding):
        latent_states = self.mapper(embedding[:, None])
        latent_states = self.final_layer_norm(latent_states)
        latent_states = self.proj_out(latent_states)
        latent_states = latent_states.view(latent_states.shape[0], self.n_dim, self.proj_size)
        return latent_states

