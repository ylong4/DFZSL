import math
import pdb
from typing import List, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

# DOWNLOAD_ROOT = '~/.cache/clip'


class AttentionPool2d(nn.Module):
    def __init__(self, sequence_len: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(sequence_len + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, n_ctx=4, ctx_init=None, deep=False):
        super().__init__()
        n_cls = len(classnames)
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        image_feature_dim = clip_model.visual.proj.shape[1]
        self.ctx_dim = ctx_dim
        self.deep = deep

        self.transformer = clip_model.transformer
        self.layers = self.transformer.layers
        self.resblocks = self.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.prompt_prefix = prompt_prefix

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        self.t2i = nn.Sequential(OrderedDict([
            ("ctx_fc", nn.Linear(ctx_dim, ctx_dim * 4)),
            ("gelu", QuickGELU()),
            ("ctx_proj", nn.Linear(ctx_dim * 4, image_feature_dim))
        ]))
        nn.init.normal_(self.t2i.ctx_fc.weight, 0.,  0.02)
        nn.init.zeros_(self.t2i.ctx_fc.bias)
        nn.init.zeros_(self.t2i.ctx_proj.weight)
        nn.init.zeros_(self.t2i.ctx_proj.bias)

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS and EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def forward(self):
        # the init will be used when computing CLIP directional loss
        image_residual = self.t2i(self.ctx).mean(dim=0)

        ctx = self.ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)


        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)


        # x.shape = [batch_size, n_ctx77, transformer.width512]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        eot = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        # pdb.set_trace()
        return eot, image_residual






class ClipPromptTuning(nn.Module):
    def __init__(self, device, classnames, residual_weight=1e-5, arch="ViT-B/16", n_ctx=4, ctx_init="a photo of a"):
        super(ClipPromptTuning, self).__init__()
        clip_model, _, _ = load(arch, device=device)#, download_root=DOWNLOAD_ROOT)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.data
        # prompt tuning
        self.prompt_learner = PromptLearner(clip_model, classnames, n_ctx, ctx_init, deep=False)
        self.residual_weight = residual_weight

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    def train(self, mode=True):
        # set train status for this class: disable all but the prompt-related modules
        if mode:
            # training:
            self.image_encoder.eval()
            self.prompt_learner.train()
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def get_image_features(self, input, image=False):
        _, image_residual = self.prompt_learner()
        if image:
            image_features = self.image_encoder(input.type(self.dtype))
        else:
            image_features = input
        image_features = image_features + self.residual_weight * image_residual
        return image_features

    def get_text_features(self):
        text_features, _ = self.prompt_learner()
        return text_features

    def get_features(self, input, image=False):
        text_features, image_residual = self.prompt_learner()

        if image:
            image_features = self.image_encoder(input.type(self.dtype))
        else:
            image_features = input

        image_features = image_features + self.residual_weight * image_residual

        return image_features, text_features


def get_vlpt(device, classnames, residual_weight, arch, n_ctx, ctx_init):

    model = ClipPromptTuning(device, classnames, residual_weight=residual_weight, arch=arch, n_ctx=n_ctx, ctx_init=ctx_init)

    return model
