import torch
from wenet.transformer.encoder import BaseEncoder
from wenet.utils.class_utils import (WENET_ACTIVATION_CLASSES,
                                     WENET_ATTENTION_CLASSES,
                                     WENET_MLP_CLASSES)
from wenet.utils.common import mask_to_bias
from wenet.utils.mask import causal_or_lookahead_mask, make_non_pad_mask


class Transformer(BaseEncoder):

    def __init__(self, config):
        super().__init__(
            n_expert=config.n_expert,
            n_expert_activated=config.n_expert_activated,
            attention_heads=config.attention_heads,
            linear_units=config.linear_units,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout_rate,
            positional_dropout_rate=config.positional_dropout_rate,
            attention_dropout_rate=config.attention_dropout_rate,
            input_layer=config.input_layer,
            pos_enc_layer_type=config.pos_enc_layer_type,
            normalize_before=config.normalize_before,
            static_chunk_size=config.static_chunk_size,
            use_dynamic_chunk=config.use_dynamic_chunk,
            global_cmvn=config.global_cmvn,
            use_dynamic_left_chunk=config.use_dynamic_left_chunk,
            query_bias=config.query_bias,
            key_bias=config.key_bias,
            value_bias=config.value_bias,
            activation_type=config.activation_type,
            gradient_checkpointing=config.gradient_checkpointing,
            use_sdpa=config.use_sdpa,
            layer_norm_type=config.layer_norm_type,
            norm_eps=config.norm_eps,
            n_kv_head=config.n_kv_head,
            head_dim=config.head_dim,
            selfattention_layer_type=config.selfattention_layer_type,
            mlp_type=config.mlp_type,
            mlp_bias=config.mlp_bias,
        )
        self.config = config

    def forward(self, mels: torch.Tensor, mels_lens: torch.Tensor):
        mask = make_non_pad_mask(mels_lens)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        att_mask = causal_or_lookahead_mask(masks, self.config.right_context,
                                            self.config.left_context)
        if self.use_sdpa:
            att_mask = mask_to_bias(att_mask, xs.dtype)
        if self.gradient_checkpointing and self.training:
            xs = self.forward_layers_checkpointed(xs, att_mask, pos_emb,
                                                  mask_pad)
        else:
            xs = self.forward_layers(xs, att_mask, pos_emb, mask_pad)
        if self.normalize_before and self.final_norm:
            xs = self.after_norm(xs)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs, masks

    def forward_chunk_by_chunk(self,
                               xs,
                               decoding_chunk_size,
                               num_decoding_left_chunks=-1):
        pass

    def forward_chunk(self, xs, offset, required_cache_size, att_cache,
                      cnn_cache, att_mask):
        pass
