# create nodes by instantiation
cogvideotextencode_30 = CogVideoTextEncode(prompt="""A golden retriever, sporting sleek black sunglasses, with its lengthy fur flowing in the breeze, sprints playfully across a rooftop terrace, recently refreshed by a light rain. The scene unfolds from a distance, the dog's energetic bounds growing larger as it approaches the camera, its tail wagging with unrestrained joy, while droplets of water glisten on the concrete behind it. The overcast sky provides a dramatic backdrop, emphasizing the vibrant golden coat of the canine as it dashes towards the viewer.  """, strength=1, force_offload=False)
cogvideotextencode_31 = CogVideoTextEncode(prompt="""""", strength=1, force_offload=True)
emptylatentimage_37 = EmptyLatentImage(width=720, height=480, batch_size=1)
cogvideosampler_35 = CogVideoSampler(num_frames=7, steps=20, cfg=6, seed=247, control_after_generate="""randomize""", scheduler="""CogVideoXDDIM""", denoise_strength=1)
cliploader_20 = CLIPLoader(clip_name="""t5xxl_fp16.safetensors""", type="""sd3""")
cogvideodecode_11 = CogVideoDecode(enable_vae_tiling=False, tile_sample_min_height=240, tile_sample_min_width=360, tile_overlap_factor_height=0.2, tile_overlap_factor_width=0.2, auto_tile_size=True)
saveanimatedwebp_38 = SaveAnimatedWEBP(filename_prefix="""ComfyUI""", fps=6, lossless=True, quality=80, method="""default""")
downloadandloadcogvideomodel_36 = DownloadAndLoadCogVideoModel(model="""THUDM/CogVideoX-5b""", precision="""bf16""", quantization="""fp8_e4m3fn""", enable_sequential_cpu_offload=False, attention_mode="""sdpa""", load_device="""main_device""")

# link nodes by invocation
latent_37 = emptylatentimage_37()
clip_20 = cliploader_20()
model_36, vae_36 = downloadandloadcogvideomodel_36(block_edit=None, lora=None, compile_args=None)
conditioning_30, clip_30 = cogvideotextencode_30(clip=clip_20)
conditioning_31, clip_31 = cogvideotextencode_31(clip=clip_30)
samples_35 = cogvideosampler_35(model=model_36, positive=conditioning_30, negative=conditioning_31, samples=latent_37, image_cond_latents=None, context_options=None, controlnet=None, tora_trajectory=None, fastercache=None, feta_args=None)
images_11 = cogvideodecode_11(vae=vae_36, samples=samples_35)
result_38 = saveanimatedwebp_38(images=images_11)
