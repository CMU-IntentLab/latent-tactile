import math
import copy
import numpy as np
import torch
import random
import wandb
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch import nn
from einops import rearrange
from tqdm import tqdm

from test_loader import SplitTrajectoryDataset
from dino_decoder import ViTImageDecoder
from dino_models import VideoTransformer, normalize_acs
from config import Args
import tyro
import json
import dataclasses

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)

    def __call__(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)


def eval_openloop(transition, decoder, img_dataloader, info, args):
    h = args.history_len
    eval_h = args.eval_len
    eval_data = next(img_dataloader)
    with torch.no_grad():
        eval_data1 = eval_data['cam_zed_embd'].to(args.device)
        img_in1 = eval_data1[[0], :h].to(args.device)

        eval_data2 =  eval_data['cam_rs_embd'].to(args.device)
        img_in2 = eval_data2[[0], :h].to(args.device)

        img_all_acs = eval_data['action'][[0]].to(args.device)
        img_all_acs = normalize_acs(img_all_acs, info, args.device)

        img_acs = eval_data['action'][[0],:h].to(args.device)
        img_acs = normalize_acs(img_acs, info, args.device)

        img_in_state = eval_data['state'][[0],:h].to(args.device)
        img_im1s = eval_data['agentview_image'][[0], :h].squeeze().to(args.device)/255.
        img_im2s = eval_data['robot0_eye_in_hand_image'][[0], :h].squeeze().to(args.device)/255.
        for k in range(eval_h-h):
            pred1, pred2, pred_state, _ = transition(img_in1, img_in2, img_in_state, img_acs)

            pred_latent = torch.cat([pred1[:,[-1]], pred2[:,[-1]]], dim=0)#.squeeze()
            pred_ims, _ = decoder(pred_latent)
            pred_ims = pred_ims.clamp(0, 1)

            pred_ims = rearrange(pred_ims, "(b t) c h w -> b t h w c", t=1)
            pred_im1, pred_im2 = torch.split(pred_ims, [img_in1.shape[0], img_in2.shape[0]], dim=0)

            
            img_im1s = torch.cat([img_im1s, pred_im1.squeeze(0)], dim=0)
            img_im2s = torch.cat([img_im2s, pred_im2.squeeze(0)], dim=0)

            # getting next inputs
            img_acs = torch.cat([img_acs[[0], 1:], img_all_acs[0,h+k].unsqueeze(0).unsqueeze(0)], dim=1)
            img_in1 = torch.cat([img_in1[[0], 1:], pred1[:, -1].unsqueeze(1)], dim=1)
            img_in2 = torch.cat([img_in2[[0], 1:], pred2[:, -1].unsqueeze(1)], dim=1)
            img_in_state = torch.cat([img_in_state[[0], 1:], pred_state[:,-1].unsqueeze(1)], dim=1)

        gt_im1 = eval_data['agentview_image'][[0], :eval_h].squeeze().to(args.device)
        gt_im2 = eval_data['robot0_eye_in_hand_image'][[0], :eval_h].squeeze().to(args.device)

        gt_imgs = torch.cat([gt_im1, gt_im2], dim=-2)/255.
        pred_imgs = torch.cat([img_im1s, img_im2s], dim=-2)
        vid = torch.cat([gt_imgs, pred_imgs], dim=-3)
        vid = vid.detach().cpu().numpy()
        vid = (vid * 255).clip(0, 255).astype(np.uint8)
        vid = rearrange(vid, "t h w c -> t c h w")
        wandb.log({"video": wandb.Video(vid, fps=20, format='mp4')})
def get_data(dataloader, info, device, image = False):
    data = next(dataloader)
    data1 = data['cam_zed_embd'].to(device)
    in1 = data1[:, :-1]
    out1 = data1[:, 1:]

    data2 =  data['cam_rs_embd'].to(device)
    in2 = data2[:, :-1]
    out2 = data2[:, 1:]

    data_state = data['state'].to(device)
    in_state = data_state[:, :-1]
    out_state = data_state[:, 1:]

    data_acs = data['action'].to(device)
    norm_acs = normalize_acs(data_acs, info, device)
    acs = norm_acs[:, :-1]

    if image:
        im1 = data['agentview_image']/255.
        im2 = data['robot0_eye_in_hand_image']/255.
        return in1, out1, in2, out2, in_state, out_state, acs, im1, im2
    return in1, out1, in2, out2, in_state, out_state, acs
def eval_tf(transition, decoder, eval_dataloader, info, args):
    h = args.history_len
    with torch.no_grad():
        in1, out1, in2, out2, in_state, out_state, acs, img1, img2 = get_data(eval_dataloader, info, args.device, image=True)
        pred1, pred2, pred_state, _ = transition(in1, in2, in_state, acs)
        pred_latent = torch.cat([pred1[:,[h-1]], pred2[:,[h-1]]], dim=0) # h-1 idx predicts timestep h
        pred_ims, _ = decoder(pred_latent)
        pred_ims = pred_ims.clamp(0, 1)
        pred_im1, pred_im2 = torch.split(pred_ims, [in1.shape[0], in2.shape[0]], dim=0)
        pred_im1 = pred_im1[0].permute(1,2,0).detach().cpu().numpy()
        pred_im2 = pred_im2[0].permute(1,2,0).detach().cpu().numpy()
        
        im1_loss = nn.MSELoss()(pred1, out1)
        im2_loss = nn.MSELoss()(pred2, out2)
        state_loss = nn.MSELoss()(pred_state, out_state)
        loss = im1_loss + im2_loss + state_loss
    img1 = img1[0, h].detach().cpu().numpy()
    img2 = img2[0, h].detach().cpu().numpy()
    wandb.log({'eval_loss': loss.item(), 'front_loss': im1_loss.item(), 'wrist_loss': im2_loss.item(), 'state_loss': state_loss.item(), 'pred_front': wandb.Image(pred_im1), 'pred_wrist': wandb.Image(pred_im2), 'front': wandb.Image(img1), 'wrist': wandb.Image(img2)})
    return loss, im1_loss, im2_loss, state_loss
    

def train_tf(wm, opt, scaler, loader, info, args):
    in1, out1, in2, out2, in_state, out_state, acs = get_data(loader, info, args.device)
    opt.zero_grad()
    # teacher forcing loss
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_amp):
        pred1, pred2, pred_state, _ = wm(in1, in2, in_state, acs)
        im1_loss_tf = nn.MSELoss()(pred1, out1)
        im2_loss_tf = nn.MSELoss()(pred2, out2)
        state_loss_tf = nn.MSELoss()(pred_state, out_state)
        loss_tf = im1_loss_tf + im2_loss_tf + state_loss_tf
    loss = loss_tf
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=1.0)
    scaler.step(opt)
    scaler.update()
    wandb.log({'train_loss': loss_tf, 'lr': opt.param_groups[0]['lr']})
    return loss_tf.item(), im1_loss_tf.item(), im2_loss_tf.item(), state_loss_tf.item()
    
def main(args):
    wandb.init(project=args.proj_name,
               name=args.exp_name, config=dataclasses.asdict(args))
    

    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    info = json.load(open(args.info_dir, 'r'))
    expert_data = SplitTrajectoryDataset(args.train_dir, args.batch_len,  num_test=0, provide_labels=False)
    expert_data_eval = SplitTrajectoryDataset(args.val_dir, args.batch_len, num_test=0, provide_labels=False)
    expert_data_imagine = SplitTrajectoryDataset(args.val_dir, args.img_len, num_test=0, provide_labels=False)
    print(len(expert_data), len(expert_data_eval), len(expert_data_imagine) )
    expert_loader = iter(DataLoader(expert_data, batch_size=args.batch_size, shuffle=True))
    expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=args.batch_size, shuffle=True))
    expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

    decoder = ViTImageDecoder().to(args.device)
    decoder.load_state_dict(torch.load(args.decoder_dir))
    decoder.eval()
    
    transition = VideoTransformer(
        image_size=(args.img_size, args.img_size),
        dim=args.dim,  # DINO feature dimension
        ac_dim=args.ac_dim,  # Action embedding dimension
        state_dim=args.state_dim,  # State dimension
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        num_frames=args.batch_len-1,
        dropout=args.dropout,
    ).to(args.device)
    #transition.load_state_dict(torch.load('/data/mobile/dinowm/testing_iter2000.pth'))
    #transition.load_state_dict(torch.load('/data/ken/dinowm_cbf/testing_iter50000.pth'))
    transition.failure_head.requires_grad_(False)
    transition.train()
    # Forward pass
    optimizer = AdamW([
        {'params': transition.transformer.parameters(), 'lr': args.base_lr},
        {'params': transition.state_head.parameters(), 'lr': args.base_lr}, 
        {'params': transition.front_head.parameters(), 'lr': args.base_lr}, 
        {'params': transition.wrist_head.parameters(), 'lr': args.base_lr}, 
        {'params': transition.action_encoder.parameters(), 'lr': args.embd_lr},
        {'params': [transition.pos_embedding], 'lr': args.embd_lr},
        {'params': [transition.temp_embedding], 'lr': args.embd_lr}
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    ema = EMA(transition, decay=0.999)

    warmup_steps = 1000
    min_lr_ratio = 0.1
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.train_iter - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    best_eval = float('inf')

    for i in tqdm(range(args.train_iter), desc="Training", unit="iter"):
        if i % len(expert_loader) == 0:
            expert_loader = iter(DataLoader(expert_data, batch_size=args.batch_size, shuffle=True))
        if i % len(expert_loader_eval) == 0:
            expert_loader_eval = iter(DataLoader(expert_data_eval, batch_size=args.batch_size, shuffle=True))
        if i % len(expert_loader_imagine) == 0:
            expert_loader_imagine = iter(DataLoader(expert_data_imagine, batch_size=1, shuffle=True))

        loss_tf, im1_loss_tf, im2_loss_tf, state_loss_tf = train_tf(transition, optimizer, scaler, expert_loader, info, args)
        scheduler.step()
        ema.update(transition)
        print(f"\rIter {i}, TF Loss: {loss_tf:.4f}, front Loss: {im1_loss_tf:.4f}, wrist Loss: {im2_loss_tf:.4f}, state Loss: {state_loss_tf:.4f}", end='', flush=True)

        # eval loop (using EMA model)
        if (i) % args.eval_iter == 0:
            eval_openloop(ema.shadow, decoder, expert_loader_imagine, info, args)
            eval_loss, im1_loss, im2_loss, state_loss = eval_tf(ema.shadow, decoder, expert_loader_eval, info, args)
            if eval_loss < best_eval:
                best_eval = eval_loss
                torch.save(ema.state_dict(), args.checkpoint_dir + 'best_testing.pth')
            torch.save(transition.state_dict(), f'{args.checkpoint_dir}testing_iter{i}.pth')
            torch.save(ema.state_dict(), f'{args.checkpoint_dir}testing_ema_iter{i}.pth')
            torch.save({
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'iter': i,
                'best_eval': best_eval,
            }, f'{args.checkpoint_dir}optim_iter{i}.pth')
            print()
            print(f"\rIter {i}, Eval Loss: {eval_loss.item():.4f}, front Loss: {im1_loss.item():.4f}, wrist Loss: {im2_loss.item():.4f}, state Loss: {state_loss.item():.4f}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)



