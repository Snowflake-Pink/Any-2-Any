# create nodes by instantiation
cliploader_20 = CLIPLoader(clip_name="""t5xxl_fp16.safetensors""", type="""sd3""")
cogvideodecode_60 = CogVideoDecode(enable_vae_tiling=True, tile_sample_min_height=240, tile_sample_min_width=360, tile_overlap_factor_height=0.2, tile_overlap_factor_width=0.2, auto_tile_size=True)
loadimage_36 = LoadImage(image="""PirateCat.png""")
cogvideotextencode_30 = CogVideoTextEncode(prompt="""a cat waving its sword""", strength=1, force_offload=False)
