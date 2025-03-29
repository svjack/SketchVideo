from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.utils import BaseOutput, logging, is_torch_version
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXBlock, CogVideoXTransformer3DModel
from diffusers.models.attention import Attention, FeedForward

from einops import rearrange, repeat


class CogVideoXPatchEmbed(nn.Module):
    """
    Patch embedding layer for CogVideoX model.
    Converts input image embeddings into patch embeddings.
    """
    def __init__(self, patch_size: int = 2, in_channels: int = 16, embed_dim: int = 1920, bias: bool = True) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=(patch_size, patch_size), stride=patch_size, bias=bias
        )

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_embeds (`torch.Tensor`):
                Input image embeddings. Expected shape: (batch_size, num_frames, channels, height, width).
        """
        batch, num_frames, channels, height, width = image_embeds.shape
        image_embeds = image_embeds.reshape(-1, channels, height, width)
        image_embeds = self.proj(image_embeds)
        image_embeds = image_embeds.view(batch, num_frames, *image_embeds.shape[1:])
        image_embeds = image_embeds.flatten(3).transpose(2, 3)  # [batch, num_frames, height x width, channels]
        image_embeds = image_embeds.flatten(1, 2)  # [batch, num_frames x height x width, channels]
        
        return image_embeds


class CrossFrameXAttnProcessor:
    """
    Processor for implementing scaled dot-product attention for the CogVideoX model.
    Applies a rotary embedding on query and key vectors without spatial normalization.
    """
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("CogVideoXAttnProcessor requires PyTorch 2.0. Upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        key_hidden_states: torch.Tensor,
        value_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        "encoder_hidden_states is not used in this processor, but is kept for compatibility with Attention class."

        # sequence_length: H * W
        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        query = attn.to_q(hidden_states)
        key = attn.to_k(key_hidden_states)
        value = attn.to_v(value_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CrossFrameBlock(nn.Module):
    """
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        
        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CrossFrameXAttnProcessor(),
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = FeedForward(
            dim,
            dim_out = dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
            mult=2,
        )

    def forward(self, hidden_states: torch.Tensor, sketch_hidden_states: torch.Tensor,
                control_frame_index: List[int], temb: torch.Tensor, num_frames: int) -> torch.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = rearrange(norm_hidden_states, 'B (T L) C -> B T L C', T=num_frames)
        
        key_norm_hidden_states = norm_hidden_states[:, control_frame_index, :, :]
        
        query_norm_hidden_states = norm_hidden_states
        query_norm_hidden_states = rearrange(query_norm_hidden_states, 'B T L C -> B (T L) C') # T = num_frames
        key_norm_hidden_states = rearrange(key_norm_hidden_states, 'B T L C -> B (T L) C') # T = 2
        
        attn_hidden_states = self.attn1(
            hidden_states=query_norm_hidden_states,
            key_hidden_states=key_norm_hidden_states,
            value_hidden_states=sketch_hidden_states,
        )
        
        hidden_states = attn_hidden_states
        
        norm_hidden_states = self.norm2(hidden_states)
        
        hidden_states = self.ff(norm_hidden_states)

        return hidden_states


class InputMappingBlock(nn.Module):
    """
    Input mapping block for processing control hidden states.
    """
    def __init__(self, dim: int, dropout: float = 0.0, activation_fn: str = "gelu-approximate",
                 final_dropout: bool = True, ff_inner_dim: Optional[int] = None, ff_bias: bool = True):
        super().__init__()

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )
        
        self.out = zero_module(nn.Linear(dim, dim))

    def forward(self, control_hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        control_hidden_states = self.ff(control_hidden_states)
        output_hidden_states = self.out(control_hidden_states)

        return output_hidden_states, control_hidden_states


@dataclass
class ControlNetOutput(BaseOutput):
    """
    The output of [`ControlNetModel`].

    Args:
        control_block_res_samples (`tuple[torch.Tensor]`):
            A tuple of downsample activations at different resolutions for each downsampling block. Each tensor should
            be of shape `(batch_size, channel * resolution, height //resolution, width // resolution)`. Output can be
            used to condition the original UNet's downsampling activations.
    """
    control_block_res_samples: Tuple[torch.Tensor]


class CogVideoControlNetModel(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    """
    CogVideoControlNetModel integrates control blocks for sketch-based video generation.
    """
    @register_to_config
    def __init__(
        self,
        control_block_index: list[int] = [0,6,12,18,24],
        num_attention_heads: int = 30,
        cross_frame_num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        time_embed_dim: int = 512,
        dropout: float = 0.0,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_bias: bool = True,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        
        # trainable copy of CogVideoX transformer blocks
        self.sketch_transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(len(control_block_index))
            ]
        )
        
        # prcoessing the sketch feature input
        self.sketch_input_mapping_blocks = nn.ModuleList(
            [
                InputMappingBlock(inner_dim)
                for _ in range(len(control_block_index))
            ]
        )
        
        # propagate the keyframe control hidden states
        self.cross_frame_blocks = nn.ModuleList(
            [
                CrossFrameBlock(
                    dim=inner_dim,
                    num_attention_heads=cross_frame_num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(len(control_block_index))
            ]
        )
        
        # output residual control features
        self.controlnet_blocks = nn.ModuleList(
            [
                zero_module(nn.Linear(inner_dim, inner_dim))
                for _ in range(len(control_block_index))
            ]
        )
        
        self.control_block_index = control_block_index
        self.control_block_dict = {}
        for i in range(len(control_block_index)):
            self.control_block_dict[control_block_index[i]] = i
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, value: bool = False):
        self.gradient_checkpointing = value

    @classmethod
    def from_transformer(cls, 
        transformer3D: CogVideoXTransformer3DModel, 
        load_weights_from_transformer: bool = True,
        control_block_index: list[int] = [0, 6, 12, 18, 24]):
        """
        Instantiate a ControlNetModel from CogVideoXTransformer3DModel.
        """
        
        controlnet = cls(
            control_block_index = control_block_index,
            num_attention_heads = transformer3D.config.num_attention_heads,
            attention_head_dim = transformer3D.config.attention_head_dim,
            time_embed_dim = transformer3D.config.time_embed_dim,
            dropout = transformer3D.config.dropout,
            norm_elementwise_affine = transformer3D.config.norm_elementwise_affine,
            norm_eps = transformer3D.config.norm_eps,
            attention_bias = transformer3D.config.attention_bias,
            activation_fn = transformer3D.config.activation_fn,
        )

        if load_weights_from_transformer:
            for i in range(len(controlnet.sketch_transformer_blocks)):
                controlnet.sketch_transformer_blocks[i].load_state_dict(transformer3D.transformer_blocks[control_block_index[i]].state_dict())
                
                controlnet.cross_frame_blocks[i] = identity_module(controlnet.cross_frame_blocks[i])
        
        return controlnet
    
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
                sketch_hidden_states: torch.Tensor, control_frame_index: List[int], num_frames: int,
                temb: torch.Tensor, image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                return_dict: bool = True, block_index: int = None):
        """
        Forward pass for CogVideoControlNetModel.
        """
        
        control_block_index = self.control_block_dict[block_index]
        
        # Mapping the sketch condition
        sketch_out_hidden_states, sketch_hidden_states = self.sketch_input_mapping_blocks[control_block_index](sketch_hidden_states)
        
        batch, _, channel = hidden_states.shape
        # [B, T, H*W, C]
        hidden_states_reshape = hidden_states.reshape(batch, num_frames, -1, channel)     
        sketch_out_hidden_states += hidden_states_reshape[:, control_frame_index, :, :].reshape(batch, -1, channel)
        
        if self.training and self.gradient_checkpointing:
            sketch_block = self.transformer_blocks[control_block_index]
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            control_hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(sketch_block),
                control_hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
            cross_frame_block = self.cross_frame_blocks[control_block_index]
            residual_features = torch.utils.checkpoint.checkpoint(
                create_custom_forward(cross_frame_block),
                control_hidden_states,
                control_frame_index,
                num_frames,
            )
            control_out_block = self.controlnet_blocks[control_block_index]
            residual_features = torch.utils.checkpoint.checkpoint(
                create_custom_forward(control_out_block),
                residual_features
            )
        else:
            # Sketch mapping 
            sketch_block = self.sketch_transformer_blocks[control_block_index]
            sketch_out_hidden_states, encoder_hidden_states = sketch_block(
                hidden_states=sketch_out_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            
            residual_features = self.cross_frame_blocks[control_block_index](hidden_states=hidden_states, 
                                                                 sketch_hidden_states=sketch_out_hidden_states, 
                                                                 control_frame_index=control_frame_index, 
                                                                 temb=temb,
                                                                 num_frames=num_frames)
            # control zero Linear
            residual_features = self.controlnet_blocks[control_block_index](residual_features)
            
        return residual_features, sketch_hidden_states


def zero_module(module: nn.Module) -> nn.Module:
    """
    Zero-initialize the parameters of a module.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def identity_module(module: nn.Module) -> nn.Module:
    """
    Initialize weights for linear layers with identity and biases with zero.
    """
    module.apply(weights_init)
    return module


def weights_init(m: nn.Module):
    """
    Initialize weights for linear layers with identity and biases with zero.
    """
    if isinstance(m, nn.Linear):
        nn.init.eye_(m.weight)
        nn.init.constant_(m.bias, 0.0)

