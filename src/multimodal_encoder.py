import copy

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
)


def _attention_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    mask = mask[:, None, None, :].to(dtype=dtype)
    return (1.0 - mask) * torch.finfo(dtype).min


class T5CoAttentionLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.text_self_attention = T5LayerSelfAttention(
            config, has_relative_attention_bias=False, layer_idx=layer_idx
        )
        self.image_self_attention = T5LayerSelfAttention(
            config, has_relative_attention_bias=False, layer_idx=layer_idx
        )
        self.text_to_image_attention = T5LayerCrossAttention(config, layer_idx=layer_idx)
        self.image_to_text_attention = T5LayerCrossAttention(config, layer_idx=layer_idx)
        self.text_ff = T5LayerFF(config)
        self.image_ff = T5LayerFF(config)

    def forward(
        self,
        text_states,
        image_states,
        text_attention_mask,
        image_attention_mask,
        text_position_bias,
        image_position_bias,
    ):
        text_states, text_position_bias = self.text_self_attention(
            text_states,
            attention_mask=text_attention_mask,
            position_bias=text_position_bias,
        )[:2]
        image_states, image_position_bias = self.image_self_attention(
            image_states,
            attention_mask=image_attention_mask,
            position_bias=image_position_bias,
        )[:2]

        text_self_states = text_states
        image_self_states = image_states
        text_states = self.text_to_image_attention(
            text_self_states,
            key_value_states=image_self_states,
            attention_mask=image_attention_mask,
        )[0]
        image_states = self.image_to_text_attention(
            image_self_states,
            key_value_states=text_self_states,
            attention_mask=text_attention_mask,
        )[0]

        return (
            self.text_ff(text_states),
            self.image_ff(image_states),
            text_position_bias,
            image_position_bias,
        )


class ViLBertT5Encoder(nn.Module):
    def __init__(self, config, shared_embedding, independent_layers=3, co_attention_layers=1):
        super().__init__()
        if independent_layers < 1 or co_attention_layers < 1:
            raise ValueError("independent_layers and co_attention_layers must both be >= 1")
        if independent_layers + co_attention_layers != config.num_layers:
            raise ValueError(
                "independent_layers + co_attention_layers must equal config.num_layers "
                f"({independent_layers} + {co_attention_layers} != {config.num_layers})"
            )

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False

        self.embed_tokens = shared_embedding
        self.dropout = nn.Dropout(encoder_config.dropout_rate)
        self.text_blocks = nn.ModuleList(
            [
                T5Block(
                    encoder_config,
                    has_relative_attention_bias=(layer_idx == 0),
                    layer_idx=layer_idx,
                )
                for layer_idx in range(independent_layers)
            ]
        )
        self.image_blocks = nn.ModuleList(
            [
                T5Block(
                    encoder_config,
                    has_relative_attention_bias=(layer_idx == 0),
                    layer_idx=layer_idx,
                )
                for layer_idx in range(independent_layers)
            ]
        )
        self.co_attention_blocks = nn.ModuleList(
            [
                T5CoAttentionLayer(encoder_config, independent_layers + layer_idx)
                for layer_idx in range(co_attention_layers)
            ]
        )
        self.text_final_layer_norm = T5LayerNorm(
            encoder_config.d_model, eps=encoder_config.layer_norm_epsilon
        )
        self.image_final_layer_norm = T5LayerNorm(
            encoder_config.d_model, eps=encoder_config.layer_norm_epsilon
        )

    @staticmethod
    def _run_independent_stream(blocks, states, attention_mask):
        position_bias = None
        for block in blocks:
            outputs = block(
                states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                use_cache=False,
                return_dict=False,
            )
            states, position_bias = outputs[:2]
        return states, position_bias

    def forward(
        self,
        text_input_ids,
        text_attention_mask,
        image_input_ids,
        image_attention_mask,
    ):
        text_states = self.dropout(self.embed_tokens(text_input_ids))
        image_states = self.dropout(self.embed_tokens(image_input_ids))
        text_mask = _attention_mask(text_attention_mask, text_states.dtype)
        image_mask = _attention_mask(image_attention_mask, image_states.dtype)

        text_states, text_position_bias = self._run_independent_stream(
            self.text_blocks, text_states, text_mask
        )
        image_states, image_position_bias = self._run_independent_stream(
            self.image_blocks, image_states, image_mask
        )

        for block in self.co_attention_blocks:
            (
                text_states,
                image_states,
                text_position_bias,
                image_position_bias,
            ) = block(
                text_states,
                image_states,
                text_mask,
                image_mask,
                text_position_bias,
                image_position_bias,
            )

        text_states = self.dropout(self.text_final_layer_norm(text_states))
        image_states = self.dropout(self.image_final_layer_norm(image_states))
        return (
            BaseModelOutput(last_hidden_state=text_states),
            BaseModelOutput(last_hidden_state=image_states),
        )
