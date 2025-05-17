from diffusers import DiffusionPipeline, AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler
from transformers import CLIPTokenizer, CLIPTextModel
import torch
torch.manual_seed(42)
from safetensors.torch import save_file, load_file
from tqdm.auto import tqdm
import numpy
from PIL import Image
import matplotlib.pyplot as plt
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import time
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

@torch.no_grad()
def test():
    model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
    # timesteps, num_inference_steps = retrieve_timesteps(
    #     model.scheduler,  num_inference_steps= 10
    # )
    # print(model.vae_scale_factor, model.default_sample_size)
    # prompt = ["A hyper realistic photo of romanian warrios who is very angry, 8K UHD, NASA photography, detailed skin pores, cinematic lighting, photorealistic"] # cowboy riding a brown cow on Mars
    # prompt = ["abstract colorful painting that can be used as background picture for a presentation, full screen coverance, really beautiful and neutral, 8K UHD, NASA photography, detailed skin pores, cinematic lighting, photorealistic"]
    # prompt = ["pleasant picture of people celebrating the end of their studies, they all seem happy and excited, 8K UHD, NASA photography, detailed skin pores, cinematic lighting, photorealistic"]
    # prompt = ["Anime-style ancient Greek warrior, comically oversized helmet with funny details (like feathers, horns, or a silly face), exaggerated armor with cute or absurd design, vibrant colors, dynamic pose, chibi or semi-realistic anime proportions, playful atmosphere, detailed background with Greek columns or olive trees, soft shading, Studio Ghibli-inspired, 4k high-quality artwork"]
    prompt = ["Ultra-realistic horror photo of a dark forest, trees with twisted faces and glowing eyes, eerie mist, hyper-detailed skin-like bark, bloodshot veins, unsettling asymmetry, cinematic lighting, 8k, photorealistic, uncanny valley, no people, pure dread"]
    # prompt = ["Warm final presentation slide, modern conference room with golden sunlight, diverse team smiling and clapping, large 'Thank You!' screen, fresh flowers on table, colorful notebooks, soft bokeh background, corporate friendly atmosphere, ultra-detailed 4k photorealistic style, inspiring positive mood"]
    print(model.check_inputs(prompt=prompt, prompt_2 = None, height = model.default_sample_size * model.vae_scale_factor, width = model.default_sample_size * model.vae_scale_factor, callback_steps=None))
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = model.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device="cpu",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )

    timesteps, num_inference_steps = retrieve_timesteps(
        model.scheduler, 80,
    )

    num_channels_latents = model.unet.config.in_channels
    latents = model.prepare_latents(
        1 * 1,
        num_channels_latents,
        256,
        256,
        prompt_embeds.dtype,
        model.device,
        torch.manual_seed(42)
    )
    extra_step_kwargs = model.prepare_extra_step_kwargs(torch.manual_seed(42), 0.)

    add_text_embeds = pooled_prompt_embeds
    if model.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = model.text_encoder_2.config.projection_dim


    add_time_ids = model._get_add_time_ids(
        (256, 256),
        (0, 0),
        (256, 256),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
    # if negative_original_size is not None and negative_target_size is not None:
    #     negative_add_time_ids = self._get_add_time_ids(
    #         negative_original_size,
    #         negative_crops_coords_top_left,
    #         negative_target_size,
    #         dtype=prompt_embeds.dtype,
    #         text_encoder_projection_dim=text_encoder_projection_dim,
    #     )
    # else:
    negative_add_time_ids = add_time_ids

    if True:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([negative_add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(model.device)
    add_text_embeds = add_text_embeds.to(model.device)
    add_time_ids = add_time_ids.to(model.device).repeat(1, 1)
    save_file({"encoder": prompt_embeds}, r"C:\study\coursework\src\trash\test_unet_encoder.safetensors")
    num_warmup_steps = max(len(timesteps) - num_inference_steps * model.scheduler.order, 0)
    timestep_cond = None
    print(timesteps)
    save_file({"sch_time":timesteps}, r"C:\study\coursework\src\trash\test_unet_timesteps.safetensors")

    save_file({"add_time_ids" : add_time_ids}, r"C:\study\coursework\src\trash\test_unet_add_time_ids.safetensors")
    save_file({"add_text_embs" : add_text_embeds}, r"C:\study\coursework\src\trash\test_unet_add_text_embeds.safetensors")
    with model.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = torch.cat([latents] * 2) 
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            # print(latent_model_input.shape, t, timestep_cond, added_cond_kwargs)
            # print(latent_model_input.shape, t, prompt_embeds.shape)
            # return
            # print(add_text_embeds.shape, add_time_ids.shape)
            # save_file({"add_time_ids" : add_time_ids}, r"C:\study\coursework\src\trash\test_unet_add_time_ids.safetensors")
            # save_file({"add_text_embs" : add_text_embeds}, r"C:\study\coursework\src\trash\test_unet_add_text_embeds.safetensors")
            # time_embeds = model.unet.add_time_proj(add_time_ids.flatten())
            # time_embeds = time_embeds.reshape((add_text_embeds.shape[0], -1))
            # add_embeds = torch.concat([add_text_embeds, time_embeds], dim=-1)
            # save_file({"l1" : model.unet.add_embedding.linear_1.weight}, r"C:\study\coursework\src\trash\test_unet_add_l1.safetensors")
            # save_file({"l1" : model.unet.add_embedding.linear_1.bias}, r"C:\study\coursework\src\trash\test_unet_add_l1_b.safetensors")
            # save_file({"l2" : model.unet.add_embedding.linear_2.weight}, r"C:\study\coursework\src\trash\test_unet_add_l2.safetensors")
            # save_file({"l2" : model.unet.add_embedding.linear_2.bias}, r"C:\study\coursework\src\trash\test_unet_add_l2_b.safetensors")
            # print(model.unet.add_embedding)
            # aug_emb = model.unet.add_embedding(add_embeds)
            # print(aug_emb, aug_emb.shape)
            # save_file({"aug_emb" : aug_emb}, r"C:\study\coursework\src\trash\test_unet_aug_emb.safetensors")
            os.remove(r"C:\study\coursework\src\trash\test_unet_input.safetensors")
            save_file({"unet_in": latent_model_input}, r"C:\study\coursework\src\trash\test_unet_input.safetensors")
            # noise_pred = model.unet(
            #     latent_model_input,
            #     t,
            #     encoder_hidden_states=prompt_embeds,
            #     timestep_cond=None, #timestep_cond = None
            #     cross_attention_kwargs=None,
            #     added_cond_kwargs=added_cond_kwargs,
            #     return_dict=False,
            # )[0]
            open("procc.flag", "w").close()
            while os.path.exists("procc.flag"):
                time.sleep(0.01)

            # if i == 0:
            #     print("\n\n\n", t)
            #     save_file({"un_out": noise_pred}, r"C:\study\coursework\src\trash\test_unet_output.safetensors")
            #     print(noise_pred)
            noise_pred = load_file(r"C:\study\coursework\src\trash\test_unet_output.safetensors")["output"]
            print("\nPython:\n", noise_pred.detach().cpu().numpy()[:10])
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)


            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % model.scheduler.order == 0):
                progress_bar.update()


    # make sure the VAE is in float32 mode, as it overflows in float16
    needs_upcasting = model.vae.dtype == torch.float16 and model.vae.config.force_upcast

    if needs_upcasting:
        model.upcast_vae()
        latents = latents.to(next(iter(model.vae.post_quant_conv.parameters())).dtype)
    elif latents.dtype != model.vae.dtype:
        if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            model.vae = model.vae.to(latents.dtype)

    # unscale/denormalize the latents
    # denormalize with the mean and std if available and not None
    has_latents_mean = hasattr(model.vae.config, "latents_mean") and model.vae.config.latents_mean is not None
    has_latents_std = hasattr(model.vae.config, "latents_std") and model.vae.config.latents_std is not None
    if has_latents_mean and has_latents_std:
        latents_mean = (
            torch.tensor(model.vae.config.latents_mean).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents_std = (
            torch.tensor(model.vae.config.latents_std).view(1, 4, 1, 1).to(latents.device, latents.dtype)
        )
        latents = latents * latents_std / model.vae.config.scaling_factor + latents_mean
    else:
        latents = latents / model.vae.config.scaling_factor

    image = model.vae.decode(latents, return_dict=False)[0]

    # cast back to fp16 if needed
    if needs_upcasting:
        model.vae.to(dtype=torch.float16)

    image = model.image_processor.postprocess(image, output_type="pil")
    model.maybe_free_model_hooks()

    image[0].save('backgroundv2.png')
    plt.imshow(image[0])
    plt.axis("off")
    plt.show()

    

test()