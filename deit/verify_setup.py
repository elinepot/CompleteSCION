#!/usr/bin/env python3
"""
Pre-flight verification script for DeiT depth-scaling experiments (plan v6.0).

IMPORTANT: This script must be run from inside the `deit_timm032/` subtree:
    $WORK/LipschitzSCION/deit/deit_timm032/

It imports the local `models.py`, `vision_transformer.py`, `scion.py` as they
exist in that directory. Running it from a different cwd will either import
the wrong files or fail.

**This script assumes `deit_timm032/models.py` has been patched to forward
`**kwargs` in the `deit_base_patch16_224` factory. The expected patch is:**

    @register_model
    def deit_base_patch16_224(pretrained=False, depth=12, alpha_skip_connection=1.0, **kwargs):
        kwargs.pop('drop_block_rate', None)   # VisionTransformer doesn't accept it
        model = VisionTransformer(
            patch_size=16, embed_dim=768, depth=depth, num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), alpha_skip_connection=alpha_skip_connection,
            **kwargs,
            )
        ...

Without this patch, step 3 will fail on the `drop_path` assertion: the factory
drops `drop_path_rate` silently and all blocks get `nn.Identity` regardless of
what you pass.

Usage:
    # Get an interactive H100 GPU session
    srun --pty --nodes=1 --ntasks-per-node=1 --cpus-per-task=24 --gres=gpu:1 \\
         -C h100 --hint=nomultithread --time=00:30:00 \\
         -A ozx@h100 --qos=qos_gpu_h100-dev bash

    # Load modules
    module purge
    module load arch/h100
    module load pytorch-gpu/py3/2.8.0

    # Silence timm 0.3.2-style import shim warnings
    export PYTHONWARNINGS="ignore::FutureWarning"

    # Run
    cd $WORK/LipschitzSCION/deit/deit_timm032
    python verify_setup.py

All 8 checks must pass before sbatch'ing the real sweep.
"""

import math
import os
import sys
import traceback
import warnings

# Silence the FutureWarnings that timm 1.x emits when deit_timm032/ imports
# from timm.models.helpers / .layers / .registry. These are shim paths that
# still work but warn on every import; they're not our problem.
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure we're running from deit_timm032/ (or at least that the expected
# local files are importable from cwd).
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

_failures = []


def _pass(msg):
    print(f"  \u2713 {msg}")


def _fail(msg):
    print(f"  \u2717 {msg}")
    _failures.append(msg)


def _info(msg):
    print(f"  {msg}")


def _section(n, title):
    print(f"\n[{n}] {title}")


print("=" * 70)
print("VERIFY SETUP: DeiT depth-scaling with Scion (plan v6.0)")
print(f"Running from: {_here}")
print("=" * 70)

# Sanity: confirm the expected local files are here. If not, we're in the
# wrong directory and everything below will fail confusingly.
for fname in ("models.py", "vision_transformer.py", "scion.py", "main_scion.py"):
    if not os.path.isfile(os.path.join(_here, fname)):
        print(f"\nERROR: {fname} not found in {_here}")
        print("Run this script from the deit_timm032/ subtree.")
        sys.exit(2)

# --------------------------------------------------------------------------
# [1] Versions
# --------------------------------------------------------------------------
_section(1, "Versions")
_info(f"Python: {sys.version.split()[0]}")
_info(f"PyTorch: {torch.__version__}")
_info(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    _info(f"CUDA version (torch): {torch.version.cuda}")
    _info(f"GPU: {torch.cuda.get_device_name(0)}")
    _info(f"GPU count visible: {torch.cuda.device_count()}")

try:
    import timm

    _info(f"timm: {timm.__version__} ({timm.__file__})")
    major = int(timm.__version__.split(".")[0])
    if major >= 1:
        _pass(
            f"timm major version {major} >= 1 (needed for torch 2.8 + shim-compat imports)"
        )
    else:
        _fail(
            f"timm {timm.__version__} is too old; need >= 1.0 (1.0.26 recommended). "
            "Install with: pip install --user --no-cache-dir timm==1.0.26"
        )
except ImportError as e:
    _fail(f"timm not importable: {e}")
    print("\nCannot continue without timm. Install with:")
    print("  pip install --user --no-cache-dir timm==1.0.26")
    sys.exit(1)

# --------------------------------------------------------------------------
# [2] timm symbol availability
#
# These are the timm symbols actually imported by files inside deit_timm032/.
# Paths like `timm.models.layers` are deprecation shims in modern timm (they
# re-export from `timm.layers` and emit FutureWarning). We check the shim
# paths specifically, because that is what the local code uses.
# --------------------------------------------------------------------------
_section(2, "timm symbols used by deit_timm032/")

symbols = [
    # vision_transformer.py
    ("timm.data", ["IMAGENET_DEFAULT_MEAN", "IMAGENET_DEFAULT_STD"]),
    ("timm.models.helpers", ["load_pretrained"]),
    ("timm.models.layers", ["DropPath", "to_2tuple", "trunc_normal_"]),
    ("timm.models.registry", ["register_model"]),
    # models.py
    ("timm.models.vision_transformer", ["_cfg"]),
    ("timm.models", ["create_model"]),
    # datasets.py
    ("timm.data.constants", ["IMAGENET_DEFAULT_MEAN", "IMAGENET_DEFAULT_STD"]),
    ("timm.data", ["create_transform"]),
    # main_scion.py
    ("timm.data", ["Mixup"]),
    ("timm.loss", ["LabelSmoothingCrossEntropy", "SoftTargetCrossEntropy"]),
    ("timm.scheduler", ["create_scheduler"]),
    ("timm.optim", ["create_optimizer"]),
    ("timm.utils", ["NativeScaler", "get_state_dict", "ModelEma", "accuracy"]),
]

all_ok = True
for module_name, names in symbols:
    try:
        mod = __import__(module_name, fromlist=names)
    except ImportError as e:
        _fail(f"cannot import module {module_name}: {e}")
        all_ok = False
        continue
    missing = [n for n in names if not hasattr(mod, n)]
    if missing:
        _fail(f"{module_name} missing: {missing}")
        all_ok = False
if all_ok:
    _pass("all required timm symbols are available")

# --------------------------------------------------------------------------
# [3] Model construction + residual factor + local-class + runtime defaults
#
# Build at L in {3, 6, 12} via the local models.py registry. Verify:
#  - block count matches --depth
#  - block.factor == 1/(2L)  (the completeP residual scaling)
#  - model is an instance of the local VisionTransformer
#  - num_classes = 1000
#  - img_size = (224, 224)
#  - **drop_path_rate is correctly threaded through the factory**: blocks[-1]
#    has a DropPath with drop_prob ≈ 0.1 (the value we build with), and
#    blocks[0] is Identity (linspace(0, 0.1, depth)[0] == 0). This is the
#    main assertion that our models.py patch is in place; if the **kwargs
#    uncomment/pop has not been applied, blocks[-1] will be Identity instead.
#  - Independent kwarg-propagation test at drop_path_rate=0.3 to confirm the
#    factory is threading the value through, not just happening to match a
#    default.
# --------------------------------------------------------------------------
_section(3, "Model construction + residual factor + kwarg propagation")

try:
    import models as _deit_models  # noqa: F401, triggers @register_model

    _pass("imported local models.py (registered deit_*patch16_224 variants)")
except Exception as e:
    _fail(f"cannot import local models.py: {e}")
    traceback.print_exc()

import vision_transformer as local_vt
from timm.models import create_model

# For the DropPath check. Import from the shim path that deit_timm032 uses.
from timm.models.layers import DropPath

# For the "not timm's VisionTransformer" check:
from timm.models.vision_transformer import VisionTransformer as timm_ViT


def _build(depth, alpha=1.0, drop_path_rate=0.1):
    # Mirror main_scion.py line 274-284 exactly, so this test exercises
    # the same create_model call shape the real runs use. Expose
    # drop_path_rate so the kwarg-propagation test below can vary it.
    return create_model(
        "deit_base_patch16_224",
        pretrained=False,
        num_classes=1000,
        drop_rate=0.0,
        drop_path_rate=drop_path_rate,
        img_size=224,
        depth=depth,
        alpha_skip_connection=alpha,
    )


def _drop_path_prob(module):
    """Return the drop probability of a DropPath, tolerant to attribute naming
    differences across timm versions."""
    for attr in ("drop_prob", "p", "drop_path_prob"):
        if hasattr(module, attr):
            return float(getattr(module, attr))
    return None


# We build every depth with drop_path_rate=0.1 (the --drop-path argparse
# default and what production runs will use).
for L in [3, 6, 12]:
    try:
        m = _build(L, drop_path_rate=0.1)
    except TypeError as e:
        if "drop_block_rate" in str(e):
            _fail(
                f"L={L}: create_model crashed on drop_block_rate. The models.py "
                f"patch is incomplete -- it must pop 'drop_block_rate' from kwargs "
                f"before forwarding (VisionTransformer doesn't accept it)."
            )
        else:
            _fail(f"L={L}: create_model failed with TypeError: {e}")
        traceback.print_exc()
        continue
    except Exception as e:
        _fail(f"L={L}: create_model failed: {e}")
        traceback.print_exc()
        continue

    # Block count
    n_blocks = len(m.blocks) if hasattr(m, "blocks") else -1
    if n_blocks != L:
        _fail(f"L={L}: model has {n_blocks} blocks, expected {L}")
        continue

    # Residual factor
    if not hasattr(m.blocks[0], "factor"):
        _fail(
            f"L={L}: block has no .factor attribute; "
            f"deit_timm032/vision_transformer.py is not the one being loaded"
        )
        continue
    expected = 1.0 / (2 * L)
    got = float(m.blocks[0].factor)
    if abs(got - expected) < 1e-8:
        _pass(f"L={L}: depth={n_blocks}, block.factor={got:.6f} (= 1/(2L) \u2713)")
    else:
        _fail(f"L={L}: block.factor={got}, expected {expected}")
        continue

    # Local VisionTransformer, not timm's
    if not isinstance(m, local_vt.VisionTransformer):
        _fail(
            f"L={L}: model class is {type(m).__name__}, expected local VisionTransformer"
        )
        continue
    if m.__class__ is timm_ViT:
        _fail(f"L={L}: model is timm's VisionTransformer, not the local one")
        continue

    # num_classes propagates
    if m.head.out_features != 1000:
        _fail(f"L={L}: head.out_features = {m.head.out_features}, expected 1000")
        continue

    # img_size propagates
    got_img = m.patch_embed.img_size
    if tuple(got_img) != (224, 224):
        _fail(f"L={L}: img_size = {got_img}, expected (224, 224)")
        continue

    # *** drop_path propagation: the main test of the models.py patch ***
    # Expected: linspace(0, 0.1, L) -- block 0 has drop_path=0 (Identity);
    # last block has drop_path=0.1 (DropPath).
    last_dp = m.blocks[-1].drop_path
    first_dp = m.blocks[0].drop_path

    if isinstance(last_dp, nn.Identity):
        _fail(
            f"L={L}: blocks[-1].drop_path is nn.Identity -- drop_path_rate did NOT "
            f"propagate through the models.py factory. The **kwargs patch is missing; "
            f"see plan's 'Local modifications to deit_timm032/' for the expected fix."
        )
        continue
    if not isinstance(last_dp, DropPath):
        _fail(
            f"L={L}: blocks[-1].drop_path is {type(last_dp).__name__}, expected DropPath"
        )
        continue

    last_prob = _drop_path_prob(last_dp)
    if last_prob is None:
        _fail(
            f"L={L}: could not read drop probability from {type(last_dp).__name__} "
            f"(tried drop_prob, p, drop_path_prob)"
        )
        continue
    if abs(last_prob - 0.1) > 1e-6:
        _fail(
            f"L={L}: blocks[-1].drop_path probability = {last_prob}, expected 0.1. "
            f"drop_path_rate is reaching the model but with the wrong value."
        )
        continue

    # block[0] should be Identity because linspace(0, 0.1, L)[0] == 0 triggers
    # the `else nn.Identity()` branch in Block.__init__
    if not isinstance(first_dp, nn.Identity):
        _fail(
            f"L={L}: blocks[0].drop_path is {type(first_dp).__name__}, "
            f"expected nn.Identity (linspace starts at 0)"
        )
        continue

    _pass(
        f"L={L}: drop_path propagation OK -- blocks[-1] is DropPath(0.1), "
        f"blocks[0] is Identity (linspace start = 0)"
    )

# Independent propagation test at a non-default value. Catches the case
# where the factory happens to hardcode 0.1 rather than threading the arg.
try:
    m_test = _build(6, drop_path_rate=0.3)
    probe_dp = m_test.blocks[-1].drop_path
    if isinstance(probe_dp, nn.Identity):
        _fail(
            "non-default kwarg test: drop_path_rate=0.3 did not propagate -- "
            "blocks[-1] is nn.Identity. The factory is still dropping the kwarg."
        )
    else:
        probe_prob = _drop_path_prob(probe_dp)
        if probe_prob is None:
            _fail(
                f"non-default kwarg test: DropPath has no readable drop_prob attribute"
            )
        elif abs(probe_prob - 0.3) < 1e-6:
            _pass(
                f"non-default kwarg test: drop_path_rate=0.3 propagated correctly "
                f"(blocks[-1].drop_prob = {probe_prob:.3f})"
            )
        else:
            _fail(
                f"non-default kwarg test: drop_path_rate=0.3 but blocks[-1] "
                f"drop_prob = {probe_prob:.3f} -- factory is not threading the value"
            )
except Exception as e:
    _fail(f"non-default kwarg test crashed: {e}")
    traceback.print_exc()

# --------------------------------------------------------------------------
# [4] modify_model regression test
#
# After modify_model(model), no nn.LayerNorm or nn.GELU instances should
# remain anywhere in the module tree. main_scion.py defines modify_model
# inline (lines 294-313); we replicate the logic here since main_scion.py
# is not importable without running distributed setup code.
# --------------------------------------------------------------------------
_section(4, "modify_model: LayerNorm \u2192 RMSNorm, GELU \u2192 \u221a2\u00b7GELU")


class _ScaledGELU(nn.Module):
    def forward(self, x):
        return (2.0**0.5) * F.gelu(x)


class _RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))


def _modify_model(module):
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, _RMSNorm())
        elif isinstance(child, nn.GELU):
            setattr(module, name, _ScaledGELU())
        else:
            _modify_model(child)


try:
    m = _build(6)
    n_ln_before = sum(1 for mm in m.modules() if isinstance(mm, nn.LayerNorm))
    n_gelu_before = sum(1 for mm in m.modules() if isinstance(mm, nn.GELU))
    _info(f"before modify_model: {n_ln_before} LayerNorm, {n_gelu_before} GELU")

    _modify_model(m)

    n_ln_after = sum(1 for mm in m.modules() if isinstance(mm, nn.LayerNorm))
    n_gelu_after = sum(1 for mm in m.modules() if isinstance(mm, nn.GELU))
    _info(f"after modify_model:  {n_ln_after} LayerNorm, {n_gelu_after} GELU")

    if n_ln_after == 0 and n_gelu_after == 0:
        _pass("no LayerNorm or GELU instances remain after modify_model")
    else:
        _fail(
            f"{n_ln_after} LayerNorm and {n_gelu_after} GELU still remain "
            f"(some are wrapped in non-leaf modules modify_model doesn't recurse into)"
        )
except Exception as e:
    _fail(f"modify_model test crashed: {e}")
    traceback.print_exc()

# --------------------------------------------------------------------------
# [5] Optimizer spec: replicate the 3-group construction from main_scion.py
#     (deit_timm032/main_scion.py lines 413-432)
# --------------------------------------------------------------------------
_section(5, "Scion optimizer groups (mirrors main_scion.py L413-432 at \u03b1=1)")

Scion = None
try:
    from scion import Scion

    _pass("imported Scion from deit_timm032/scion.py")
except Exception as e:
    _fail(f"cannot import Scion: {e}")
    traceback.print_exc()

if Scion is not None:
    try:
        L = 3
        alpha = 1.0
        m = _build(L, alpha=alpha)
        _modify_model(m)

        # Replicate the partition logic from main_scion.py L399-408
        all_params = list(m.parameters())
        head_params = list(m.head.parameters())
        head_weights = [p for p in head_params if p.dim() > 1]
        pos_embed_param = m.pos_embed
        cls_token_param = m.cls_token

        other_params = [
            p for p in all_params if id(p) not in {id(lp) for lp in head_weights}
        ]
        other_params = [
            p
            for p in other_params
            if id(p) not in {id(pos_embed_param), id(cls_token_param)}
        ]

        lr_factor = (2 * L) ** (alpha - 1)  # = 1 when alpha=1
        SCALE = 25  # a Phase 1 grid point
        LAST_SCALE = 500  # fixed throughout

        optim_groups = [
            {
                "params": other_params,
                "norm": "Auto",
                "norm_kwargs": {},
                "scale": SCALE,
                "lr_scale": lr_factor,
            },
            {
                "params": head_weights,
                "norm": "Sign",
                "norm_kwargs": {},
                "scale": LAST_SCALE,
                "lr_scale": 1,
            },
            {
                "params": [pos_embed_param, cls_token_param],
                "norm": "BiasRMS",
                "norm_kwargs": {},
                "scale": SCALE,
                "lr_scale": 1,
            },
        ]

        n_tr = sum(p.numel() for p in other_params)
        n_hw = sum(p.numel() for p in head_weights)
        n_pc = pos_embed_param.numel() + cls_token_param.numel()
        _info(
            f"group 0 (trunk/Auto):               {len(other_params)} tensors, "
            f"{n_tr:>11,} params, scale={SCALE}, lr_scale={lr_factor}"
        )
        _info(
            f"group 1 (head_weights/Sign):        {len(head_weights)} tensors, "
            f"{n_hw:>11,} params, scale={LAST_SCALE}, lr_scale=1"
        )
        _info(
            f"group 2 (pos_embed+cls_token/BiasRMS): 2 tensors, "
            f"{n_pc:>11,} params, scale={SCALE}, lr_scale=1"
        )

        # Every parameter in the model should be in exactly one group
        all_ids = {id(p) for p in all_params}
        grouped_ids = set()
        dup = False
        for g in optim_groups:
            for p in g["params"]:
                if id(p) in grouped_ids:
                    _fail(f"parameter with id {id(p)} appears in multiple groups")
                    dup = True
                grouped_ids.add(id(p))
        missing = all_ids - grouped_ids
        extra = grouped_ids - all_ids
        if not dup and not missing and not extra:
            _pass(f"all {len(all_ids)} parameter tensors partitioned into 3 groups")
        elif missing or extra:
            _fail(f"partition incomplete: {len(missing)} missing, {len(extra)} extra")

        # alpha=1 sanity
        if abs(lr_factor - 1.0) < 1e-12:
            _pass("lr_scale == 1 for alpha=1 (the plan's only supported setting)")
        else:
            _fail(f"lr_scale = {lr_factor} != 1; are you running with alpha != 1?")

        # Instantiate Scion (don't step yet — step() probe is in [6], real step in [8])
        _ = Scion(optim_groups, lr=2**-13.5, momentum=0.1)
        _pass(
            f"Scion instantiated with 3 groups, lr=2^-13.5 ({2**-13.5:.3e}), momentum=0.1"
        )

    except Exception as e:
        _fail(f"Scion construction failed: {e}")
        traceback.print_exc()

# --------------------------------------------------------------------------
# [6] lr_scale regression check
#
# deit_timm032/scion.py applies `lr = lr * group["lr_scale"]` in step() (line
# 279). This is the reference behavior. The outer deit/scion.py DROPS this
# line, which is inert at alpha=1 but wrong at alpha != 1. We should be on
# the reference behavior.
#
# Expectation: step magnitude matches lr * lr_scale (not just lr).
# --------------------------------------------------------------------------
_section(6, "lr_scale behavior in deit_timm032/scion.py (expect: applied)")

if Scion is not None:
    try:
        # BiasRMS.lmo on [1,-1,1,-1] returns it unchanged (rms = 1), so the
        # step magnitude per element is lr_effective * scale * 1. With
        # scale=1 and momentum=1 (bypasses EMA):
        #   |delta p| == lr * lr_scale    (reference behavior in deit_timm032)
        #   |delta p| == lr               (if someone removed the lr_scale line)
        #
        # Note: we're using unconstrained=True here to isolate the step magnitude
        # from the `p.data.mul_(1-lr)` shrinkage that the constrained form applies.
        # Scion's `unconstrained` defaults to False; override via the group dict.
        p = nn.Parameter(torch.zeros(4))
        LR = 0.1
        LR_SCALE = 7.0
        opt = Scion(
            [
                {
                    "params": [p],
                    "norm": "BiasRMS",
                    "norm_kwargs": {},
                    "scale": 1.0,
                    "lr_scale": LR_SCALE,
                    "unconstrained": True,
                }
            ],
            lr=LR,
            momentum=1.0,
        )
        p.grad = torch.tensor([1.0, -1.0, 1.0, -1.0])
        before = p.data.clone()
        opt.step()
        step = (p.data - before).abs().max().item()

        expected_ref = LR * LR_SCALE  # 0.7
        expected_buggy = LR  # 0.1

        if abs(step - expected_ref) < 1e-5:
            _pass(
                f"|step|={step:.6f} matches lr * lr_scale = {expected_ref} "
                f"(reference behavior in deit_timm032/scion.py \u2713)"
            )
        elif abs(step - expected_buggy) < 1e-6:
            _fail(
                f"|step|={step:.6f} matches lr ({expected_buggy}) but NOT lr*lr_scale "
                f'({expected_ref}). Someone has removed the `lr = lr * group["lr_scale"]` '
                f"line from scion.py. At alpha=1 this plan still works (lr_scale=1), "
                f"but the regression means alpha != 1 runs would silently disagree "
                f"with their intended LR. Restore the line before proceeding."
            )
        else:
            _fail(
                f"|step|={step:.6f} matches neither lr ({expected_buggy}) nor "
                f"lr*lr_scale ({expected_ref}). Unexpected; investigate scion.py."
            )
    except Exception as e:
        _fail(f"lr_scale probe crashed: {e}")
        traceback.print_exc()

# --------------------------------------------------------------------------
# [7] ImageNet dataset path discovery
# --------------------------------------------------------------------------
_section(7, "ImageNet path discovery")

dsdir = os.environ.get("DSDIR", "")
_info(f"$DSDIR = {dsdir if dsdir else '(not set)'}")

candidates = []
if dsdir:
    candidates += [
        os.path.join(dsdir, "imagenet"),
        os.path.join(dsdir, "ImageNet"),
    ]
candidates += [
    "/gpfsdswork/dataset/imagenet",
    "/gpfsdswork/dataset/ImageNet",
]

found_path = None
for p in candidates:
    if os.path.isdir(p):
        has_train = os.path.isdir(os.path.join(p, "train"))
        has_val = os.path.isdir(os.path.join(p, "val"))
        _info(f"{p}: exists (train/={has_train}, val/={has_val})")
        if has_train and has_val and found_path is None:
            found_path = p
    else:
        _info(f"{p}: not found")

if found_path:
    train_dir = os.path.join(found_path, "train")
    val_dir = os.path.join(found_path, "val")
    try:
        n_classes_train = sum(
            1
            for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        )
        n_classes_val = sum(
            1 for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))
        )
        _info(f"{train_dir}: {n_classes_train} class directories")
        _info(f"{val_dir}:   {n_classes_val} class directories")
        if n_classes_train == 1000 and n_classes_val == 1000:
            _pass(f"ImageNet found at {found_path} (train/val both have 1000 classes)")
        else:
            _fail(
                f"ImageNet at {found_path}: train={n_classes_train}, val={n_classes_val}, "
                f"expected 1000 each. If val/ is a flat directory of images (not class "
                f"subdirs), datasets.py's ImageFolder call will fail -- val/ must be "
                f"organized into 1000 subdirectories."
            )
    except OSError as e:
        _fail(f"cannot list {train_dir} or {val_dir}: {e}")
else:
    _fail(
        "no ImageNet directory with train/ and val/ found in candidates. "
        "Set --data-path explicitly or symlink the ImageNet mirror."
    )

# --------------------------------------------------------------------------
# [8] Forward/backward/step under fp16 autocast + NativeScaler
#
# This matches what engine.py actually does: torch.amp.autocast('cuda')
# (default dtype fp16) combined with timm's NativeScaler (= GradScaler).
# Uses scale=25, last_scale=500, lr=2^-13.5, L=3 -- a Phase 1 grid point.
#
# Caveat on step magnitude: with world_size=1 here and world_size=16 in the
# real runs, main_scion.py's linear_scaled_lr would scale differently. We
# are passing a raw lr to Scion directly (no batch-size scaling), so the
# step magnitude here corresponds to the pre-scaled grid value, not what
# the sweep runs actually see. That's fine for a smoke test.
# --------------------------------------------------------------------------
_section(
    8, "Forward/backward/step (fp16 autocast + NativeScaler, scale=25, lr=2^-13.5, L=3)"
)

if not torch.cuda.is_available():
    _fail("no CUDA device; cannot run GPU step")
elif Scion is None:
    _fail("Scion not available; skipping")
else:
    try:
        from timm.utils import NativeScaler
    except ImportError as e:
        _fail(f"cannot import NativeScaler: {e}")
        NativeScaler = None

    if NativeScaler is not None:
        try:
            L = 3
            m = _build(L, alpha=1.0)
            _modify_model(m)
            m = m.cuda()

            all_params = list(m.parameters())
            head_params = list(m.head.parameters())
            head_weights = [p for p in head_params if p.dim() > 1]
            pos_embed_param = m.pos_embed
            cls_token_param = m.cls_token
            other_params = [
                p for p in all_params if id(p) not in {id(lp) for lp in head_weights}
            ]
            other_params = [
                p
                for p in other_params
                if id(p) not in {id(pos_embed_param), id(cls_token_param)}
            ]

            optim_groups = [
                {
                    "params": other_params,
                    "norm": "Auto",
                    "norm_kwargs": {},
                    "scale": 25,
                    "lr_scale": 1,
                },
                {
                    "params": head_weights,
                    "norm": "Sign",
                    "norm_kwargs": {},
                    "scale": 500,
                    "lr_scale": 1,
                },
                {
                    "params": [pos_embed_param, cls_token_param],
                    "norm": "BiasRMS",
                    "norm_kwargs": {},
                    "scale": 25,
                    "lr_scale": 1,
                },
            ]
            opt = Scion(optim_groups, lr=2**-13.5, momentum=0.1)
            loss_scaler = NativeScaler()

            # Synthetic batch -- batch size 4 to keep this quick on dev QoS.
            x = torch.randn(4, 3, 224, 224, device="cuda")
            y = torch.randint(0, 1000, (4,), device="cuda")

            # Snapshot a trunk parameter to verify change
            p_ref = other_params[0]
            before = p_ref.data.clone()

            # fp16 autocast, matching engine.py line 47
            with torch.amp.autocast("cuda"):
                logits = m(x)
                loss = F.cross_entropy(logits, y)

            opt.zero_grad()
            # NativeScaler handles backward + unscale + optimizer.step()
            loss_scaler(
                loss, opt, clip_grad=None, parameters=m.parameters(), create_graph=False
            )

            step_mag = (p_ref.data - before).abs().max().item()
            peak_mib = torch.cuda.max_memory_allocated() // 1024 // 1024

            _info(f"loss: {loss.item():.4f}")
            _info(f"peak GPU memory: {peak_mib} MiB")
            _info(f"max |delta p| on a trunk parameter: {step_mag:.3e}")

            ok = True
            if not math.isfinite(loss.item()):
                _fail(f"loss is not finite: {loss.item()}")
                ok = False
            # cross_entropy on 1000 classes with untrained weights: expect ~log(1000) = 6.9
            # Generous bounds to allow for fp16 noise and the √2·GELU / RMSNorm reshaping.
            if not (3.0 <= loss.item() <= 20.0):
                _fail(
                    f"loss {loss.item():.4f} outside sane range [3, 20] for an "
                    f"untrained DeiT on random labels; model may be misconfigured"
                )
                ok = False
            if step_mag == 0:
                _fail(
                    "parameter did not change after opt.step(); gradients may be zero "
                    "(possibly blocked by NativeScaler's inf-check?)"
                )
                ok = False
            elif step_mag > 1.0:
                _fail(
                    f"step magnitude {step_mag:.3e} is implausibly large (>1); "
                    f"optimizer is likely misconfigured"
                )
                ok = False

            if ok:
                _pass(
                    "one training step completed, loss finite & in range, parameters updated"
                )
        except Exception as e:
            _fail(f"step crashed: {e}")
            traceback.print_exc()

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "=" * 70)
if _failures:
    print(f"FAILED: {len(_failures)} issue(s)")
    for f in _failures:
        print(f"  - {f}")
    print("=" * 70)
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
    print("=" * 70)
    print("\nReady to submit the Phase 1 sweep.")
    sys.exit(0)
