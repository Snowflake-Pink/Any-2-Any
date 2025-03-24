# create nodes by instantiation
cliploader_20 = CLIPLoader(clip_name="""t5xxl_fp16.safetensors""", type="""sd3""")
cogvideodecode_60 = CogVideoDecode(enable_vae_tiling=True, tile_sample_min_height=240, tile_sample_min_width=360, tile_overlap_factor_height=0.2, tile_overlap_factor_width=0.2, auto_tile_size=True)
loadimage_36 = LoadImage(image="""PirateCat.png""")
cogvideotextencode_30 = CogVideoTextEncode(prompt="""a cat waving its sword""", strength=1, force_offload=False)
cogvideotextencode_31 = CogVideoTextEncode(prompt="""The video is not of a high quality, it has a low resolution. Watermark present in each frame. Strange motion trajectory. """, strength=1, force_offload=True)
cogvideoimageencode_62 = CogVideoImageEncode(enable_tiling=False, noise_aug_strength=0, strength=1, start_percent=0, end_percent=1)
cogvideosampler_63 = CogVideoSampler(num_frames=10, steps=40, cfg=6, seed=200365243842158, control_after_generate="""randomize""", scheduler="""CogVideoXDDIM""", denoise_strength=1)
saveanimatedwebp_64 = SaveAnimatedWEBP(filename_prefix="""ComfyUI""", fps=7, lossless=True, quality=100, method="""default""")
downloadandloadcogvideomodel_59 = DownloadAndLoadCogVideoModel(model="""kijai/CogVideoX-5b-1.5-I2V""", precision="""bf16""", quantization="""fp8_e4m3fn""", enable_sequential_cpu_offload=False, attention_mode="""sdpa""", load_device="""main_device""")

