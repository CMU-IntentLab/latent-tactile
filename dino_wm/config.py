from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Args:
    proj_name: Optional[str] = "dino-WM"
    exp_name: Optional[str] = "WM"
    """the name of this experiment"""
    seed: Optional[int] = 0
    device: Optional[str] = "cuda:0"

    batch_size: Optional[int] = 16 #16
    batch_len: Optional[int] = 4
    eval_len: Optional[int] = 32 #16
    history_len: Optional[int] = 3
    img_len: Optional[int] = 32

    train_dir: Optional[str] = "/data/mobile/train/buffer.h5"
    train_label_dir: Optional[str] = "/data/mobile/train_labeled/buffer_v2.h5"
    brt_label_dir: Optional[str] = "/data/mobile/train_labeled/buffer_v2.h5"#unsafe_labeled/buffer.h5"
    val_dir: Optional[str] = "/data/mobile/unsafe_labeled_test/buffer_v2.h5"
    val_label_dir: Optional[str] = "/data/mobile/unsafe_labeled_test/buffer_v2.h5"
    info_dir: Optional[str] = "/data/mobile/train/info.json"

    decoder_dir: Optional[str] = "/data/mobile/dinowm/testing_decoder.pth"
    base_wm_dir: Optional[str] = "/data/mobile/dinowm/testing_ema_iter40000.pth" #testing_iter85000.pth"
    wm_dir: Optional[str] = "/data/mobile/dinowm/classifier_last_chance999.pth" #2749.pth" #classifier_gp_0022499.pth"
    brt_dir: Optional[str] = "/home/intent/Projects/latent-cbf/logs/dinowm_demo/epoch_id_3/rotvec_policy.pth"
    checkpoint_dir: Optional[str] = "/data/mobile/dinowm/"
    # 749 51 37 5 52 / 55 29 1 58
    # 2499 41 8 27 67
    # ViT parameters
    img_size: Optional[int] = 224
    dim: Optional[int] = 384
    ac_dim: Optional[int] = 10
    state_dim: Optional[int] = 8
    depth: Optional[int] = 6
    heads: Optional[int] = 16
    mlp_dim: Optional[int] = 2048
    dropout: Optional[float] = 0.1
    
    base_lr: Optional[float] = 5e-5
    embd_lr: Optional[float] = 5e-4
    decoder_lr: Optional[float] = 2e-4
    dino_disc_ckpt: Optional[str] = "/data/mobile/dinowm/dino_deitsmall8_pretrain.pth"
    train_iter: Optional[int] = 50000
    eval_iter: Optional[int] = 1000
    use_amp: Optional[bool] = True
    
    margin: float = 0.75
    no_gp: bool = False #True
    control_net: List[int] = field(default_factory=lambda: [512, 512, 512,512]) # type=int, nargs="*", default=None) # for control policy
    critic_net: List[int] = field(default_factory=lambda: [512, 512, 512,512])  # type=int, nargs="*", default=None) # for critic net
    residual_critic: bool = False
    epo: int = 15
    traj: int = 1

    w_zs: float = 0.2
    w_gp: float = 10.
    w_relu: float = 100.
    gp_thresh: float = 0.02
    
    folder: str = '/data'
    mode: str = 'cbf'
