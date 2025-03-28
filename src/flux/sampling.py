import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "y": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1.0,
    neg_ip_scale: Tensor | float = 1.0
):
    i = 0
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale, 
        )
        if i >= timestep_to_start_cfg:
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec, 
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img

def denoise_controlnet(
    model: Flux,
    controlnet:None,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=0.7,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
):
    # this is ignored for schnell
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=controlnet_cond,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        # if i >= timestep_to_start_cfg:
        if False:
            neg_block_res_samples = controlnet(
                        img=img,
                        img_ids=img_ids,
                        controlnet_cond=controlnet_cond,
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        guidance=guidance_vec,
                    )
            neg_pred = model(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[i * controlnet_gs for i in neg_block_res_samples],
                image_proj=neg_image_proj,
                ip_scale=neg_ip_scale, 
            )     
            pred = neg_pred + true_gs * (pred - neg_pred)
   
        img = img + (t_prev - t_curr) * pred

        i += 1
    return img

def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def denoise_controlnet_mix(
    model: Flux,
    controlnet:None, 
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs = 1,
    controlnet_gs=1,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    neg_image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
    neg_ip_scale: Tensor | float = 1, 
    neg_txt: Tensor=None,
    neg_txt_ids: Tensor=None,
    neg_vec: Tensor=None,
):
    # this is ignored for schnell
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    depth,normal,hand,seg=controlnet_cond
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        latent=torch.cat([depth,normal,hand,seg],dim=1)
        block_res_samples = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=latent,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        controlnet_gs=1
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i * controlnet_gs for i in block_res_samples],
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img



def denoise_double_control(
    model: Flux,
    controlnet1, 
    controlnet2,
    # model input
    inp_condition_true,
    inp_condition_default,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    control_gs=(1.0,1.0,0,0),
):
    # this is ignored for schnell
    i = 0
    img=inp_condition_true['img']
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    depth,normal,hand,seg=controlnet_cond
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        physical_latent=normal
        segmantic_latent=hand
        default_latent=-torch.ones_like(segmantic_latent,device=segmantic_latent.device,dtype=segmantic_latent.dtype)
        inp_condition_true['img']=img
        inp_condition_default['img']=img
        block_res_samples_1 = controlnet1(
                    **inp_condition_true,
                    controlnet_cond=physical_latent,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        default_res_1=controlnet1(
                    controlnet_cond=default_latent,
                    **inp_condition_default,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        block_res_samples_2=controlnet2(
                    controlnet_cond=segmantic_latent,
                    **inp_condition_true,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        default_res_2=controlnet2(
                    controlnet_cond=default_latent,
                    **inp_condition_default,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        pred = model(
            **inp_condition_true,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i*control_gs[0]+j*control_gs[1]+k*control_gs[2]+l*control_gs[3] for i,j,k,l in zip(block_res_samples_1,block_res_samples_2,default_res_1,default_res_2)],
        )
        
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img

def denoise_single_control(
    model: Flux,
    controlnet:None, 
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    control_gs=(1.0,0),
    **kwargs
):
    # this is ignored for schnell
    i = 0
    use_type:tuple=kwargs.get('use_type',(1,1,1,1))
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    depth,normal,hand,seg=controlnet_cond
    if use_type[0]==0:
        depth=-torch.ones_like(depth,device=depth.device,dtype=depth.dtype)
    if use_type[1]==0:
        normal=-torch.ones_like(depth,device=depth.device,dtype=depth.dtype)
    if use_type[2]==0:
        hand=-100*torch.ones_like(depth,device=depth.device,dtype=depth.dtype)
    if use_type[3]==0:
        seg=-100*torch.ones_like(depth,device=depth.device,dtype=depth.dtype)
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        latent=torch.cat([depth,normal,hand,seg],dim=1)
        default_latent=-torch.ones_like(latent,device=latent.device,dtype=latent.dtype)
        block_res_samples_1 = controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=latent,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )
        default_res_1=controlnet(
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=default_latent,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
        )
        pred = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            block_controlnet_hidden_states=[i*control_gs[0]+j*control_gs[1] for i,j in zip(block_res_samples_1,default_res_1)],
        )
        
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img
def denoise_full_control(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # ip-adapter parameters
    image_proj: Tensor=None, 
    ip_scale: Tensor | float = 1, 
):
    # this is ignored for schnellc
    i = 0
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    depth,normal,hand,seg=controlnet_cond
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        latent=torch.cat([img,depth,normal,hand,seg],dim=2)
        pred = model(
            img=latent,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            image_proj=image_proj,
            ip_scale=ip_scale,
        )
        img = img + (t_prev - t_curr) * pred
        i += 1
    return img