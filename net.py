from timm.models.efficientnet import default_cfgs, EfficientNet
from timm.models.efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights,\
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from timm.models.helpers import build_model_with_cfg
from attacker import PGDAttacker, NoOpAttacker
from timm.models.registry import register_model
from timm.models import create_model
from functools import partial
import torch.nn as nn
import torch
def to_status(m, status):
    if hasattr(m, 'batch_type'):
        m.batch_type = status


to_clean_status = partial(to_status, status='clean')
to_adv_status = partial(to_status, status='adv')
to_mix_status = partial(to_status, status='mix')


class AdvEfficientNet(EfficientNet):
    def __init__(self, block_args, num_classes=1000, num_features=1280, in_chans=3, stem_size=32, fix_stem=False,
                 output_stride=32, pad_type='', round_chs_fn=round_channels, act_layer=None, norm_layer=None,
                 se_layer=None, drop_rate=0., drop_path_rate=0., global_pool='avg', attacker=NoOpAttacker()):
        print('AdvEfficientNet', norm_layer)
        super().__init__(block_args, num_classes, num_features, in_chans, stem_size, fix_stem, output_stride, pad_type, round_chs_fn, act_layer, norm_layer, se_layer, drop_rate, drop_path_rate, global_pool)
        self.attacker = attacker 
        self.mixbn = False
    def set_attacker(self, attacker):
        self.attacker = attacker
    
    def set_mixbn(self, mixbn):
        self.mixbn = mixbn
    
    def forward(self, x, labels):
        training = self.training
        input_len = len(x)
        # only during training do we need to attack, and cat the clean and auxiliary pics
        if training:
            self.eval()
            self.apply(to_adv_status)
            if isinstance(self.attacker, NoOpAttacker):
                images = x
                targets = labels 
            else:
                aux_images, _ = self.attacker.attack(x, labels, self._forward_impl)
                images = torch.cat([x, aux_images], dim=0)
                targets = torch.cat([labels, labels], dim=0)
            self.train()
            if self.mixbn:
                # the DataParallel usually cat the outputs along the first dimension simply,
                # so if we don't change the dimensions, the outputs will be something like
                # [clean_batches_gpu1, adv_batches_gpu1, clean_batches_gpu2, adv_batches_gpu2...]
                # Then it will be hard to distinguish clean batches and adversarial batches.
                self.apply(to_mix_status)
                return self._forward_impl(images).view(2, input_len, -1).transpose(1, 0), targets.view(2, input_len).transpose(1, 0)
            else:
                self.apply(to_clean_status)
                return self._forward_impl(images), targets
        else:
            images = x
            targets = labels
            return self._forward_impl(images), targets
        
def _create_effnet_adv(variant, pretrained=False, **kwargs):
    model_cls = AdvEfficientNet
    model = build_model_with_cfg(
        model_cls, variant, pretrained,
        default_cfg=default_cfgs[variant],
        pretrained_strict=True,
        kwargs_filter=False,
        **kwargs)
    return model

def _gen_efficientnet_adv(variant, channel_multiplier=1.0, depth_multiplier=1.0, pretrained=False, **kwargs):
    arch_def = [
        ['ds_r1_k3_s1_e1_c16_se0.25'],
        ['ir_r2_k3_s2_e6_c24_se0.25'],
        ['ir_r2_k5_s2_e6_c40_se0.25'],
        ['ir_r3_k3_s2_e6_c80_se0.25'],
        ['ir_r3_k5_s1_e6_c112_se0.25'],
        ['ir_r4_k5_s2_e6_c192_se0.25'],
        ['ir_r1_k3_s1_e6_c320_se0.25'],
    ]
    round_chs_fn = partial(round_channels, multiplier=channel_multiplier)
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, depth_multiplier),
        num_features=round_chs_fn(1280),
        stem_size=32,
        round_chs_fn=round_chs_fn,
        act_layer=resolve_act_layer(kwargs, 'swish'),
        norm_layer=kwargs.pop('norm_layer', None) or partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        **kwargs,
    )
    model = _create_effnet_adv(variant, pretrained, **model_kwargs)
    return model

@register_model
def covid_effnet(pretrained=False, **kwargs):
    model = _gen_efficientnet_adv(
        'covid_effnet', channel_multiplier=1.6, depth_multiplier=2.2, pretrained=pretrained, **kwargs)
    return model