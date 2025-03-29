from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from einops import rearrange, repeat


def add_method(cls):
    """
    Decorator to dynamically add a method to a class.
    """
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator


@add_method(CogVideoXTransformer3DModel)
def get_embedding(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for the input hidden states and encoder hidden states.
    """
    batch_size, num_frames, channels, height, width = hidden_states.shape

    # Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

    # Position embedding
    text_seq_length = encoder_hidden_states.shape[1]
    if not self.config.use_rotary_positional_embeddings:
        seq_length = height * width * num_frames // (self.config.patch_size**2)
        pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
        hidden_states += pos_embeds
        hidden_states = self.embedding_dropout(hidden_states)

    encoder_hidden_states = hidden_states[:, :text_seq_length]
    hidden_states = hidden_states[:, text_seq_length:]

    return emb, encoder_hidden_states, hidden_states


@add_method(CogVideoXTransformer3DModel)
def get_embedding_frame(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    control_frame_index: Tuple[int],
    timestep: Union[int, float, torch.LongTensor],
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute embeddings for specific frames based on control frame indices.
    """
    max_num_frames = 13
    batch_size, num_frames, channels, height, width = hidden_states.shape

    # Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # Patch embedding
    hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)

    # Position embedding
    text_seq_length = encoder_hidden_states.shape[1]
    if not self.config.use_rotary_positional_embeddings:
        seq_length = height * width * max_num_frames // (self.config.patch_size**2)
        # Get Text and image embedding
        pos_embeds = self.pos_embedding[:, : text_seq_length + seq_length]
        # Split text embedding
        encoder_pos_embeds = pos_embeds[:, :text_seq_length]
        image_pos_embeds = rearrange(pos_embeds[:, text_seq_length:], 'B (T L) C -> B T L C', T=max_num_frames)

        # Process each batch independently
        image_pos_embeds_list = [
            image_pos_embeds[0, control_frame_index[i]] for i in range(batch_size)
        ]
        image_pos_embeds = torch.stack(image_pos_embeds_list)
        image_pos_embeds = rearrange(image_pos_embeds, 'B T L C -> B (T L) C')

        # Add embeddings
        encoder_hidden_states = hidden_states[:, :text_seq_length] + encoder_pos_embeds
        hidden_states = hidden_states[:, text_seq_length:] + image_pos_embeds

        # Apply dropout
        encoder_hidden_states = self.embedding_dropout(encoder_hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)
    else:
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

    return emb, encoder_hidden_states, hidden_states


@add_method(CogVideoXTransformer3DModel)
def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: Union[int, float, torch.LongTensor],
    controlNet: torch.nn.Module,
    control_frame_index: Tuple[int],
    sketch_mask: torch.Tensor,
    image_mask: torch.Tensor,
    sketch_hidden_states: Optional[Tuple[torch.Tensor]] = None,
    image_hidden_states: Optional[Tuple[torch.Tensor]] = None,
    timestep_cond: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_dict: bool = True,
    conditioning_scale: float = 1.0,
    height: int = 60,
    width: int = 90,
    channels: int = 16,
) -> Union[Transformer2DModelOutput, Tuple[torch.Tensor]]:
    """
    Forward pass for the transformer model with ControlNet integration.
    """
    batch_size, num_frames, _ = hidden_states.shape
    num_frames = int(num_frames / (height * width / self.config.patch_size / self.config.patch_size))
    text_seq_length = encoder_hidden_states.shape[1]

    # Time embedding
    timesteps = timestep
    t_emb = self.time_proj(timesteps)

    # timesteps does not contain any weights and will always return f32 tensors
    # but time_embedding might actually be running in fp16. so we need to cast here.
    # there might be better ways to encapsulate this.
    t_emb = t_emb.to(dtype=hidden_states.dtype)
    emb = self.time_embedding(t_emb, timestep_cond)

    # Get ControlNet block index
    control_block_index = (
        controlNet.module.control_block_index if isinstance(controlNet, torch.nn.parallel.DistributedDataParallel)
        else controlNet.control_block_index
    )

    # Transformer blocks
    for i, block in enumerate(self.transformer_blocks):
        if i in control_block_index:
            residual_features, sketch_hidden_states, image_hidden_states = controlNet(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                sketch_hidden_states=sketch_hidden_states,
                image_hidden_states=image_hidden_states,
                sketch_mask=sketch_mask,
                image_mask=image_mask,
                control_frame_index=control_frame_index,
                num_frames=num_frames,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                block_index=i,
            )

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            hidden_states, encoder_hidden_states = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                hidden_states,
                encoder_hidden_states,
                emb,
                image_rotary_emb,
                **ckpt_kwargs,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

        # Add Control residual features
        if i in control_block_index:
            hidden_states += residual_features * conditioning_scale

    # Final normalization and projection
    if not self.config.use_rotary_positional_embeddings:
        # CogVideoX-2B
        hidden_states = self.norm_final(hidden_states)
    else:
        # CogVideoX-5B
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        hidden_states = self.norm_final(hidden_states)
        hidden_states = hidden_states[:, text_seq_length:]

    # Final block
    hidden_states = self.norm_out(hidden_states, temb=emb)
    hidden_states = self.proj_out(hidden_states)

    # Unpatchify
    p = self.config.patch_size
    output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, channels, p, p)
    output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)

