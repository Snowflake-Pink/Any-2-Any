# create nodes by instantiation
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""majicmixRealistic_v7.safetensors""")
cliptextencode_6 = CLIPTextEncode(text="""1girl, """)
cliptextencode_7 = CLIPTextEncode(text="""lowres,zombie,horror,nsfw, """)
facedetailer_11 = FaceDetailer(guide_size=384, guide_size_for=True, max_size=1024, seed=266448747412199, control_after_generate="""randomize""", steps=20, cfg=4, sampler_name="""euler_ancestral""", scheduler="""normal""", denoise=0.5, feather=5, noise_mask=True, force_inpaint=True, bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3, sam_detection_hint="""center-1""", sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0, sam_mask_hint_threshold=0.7000000000000001, sam_mask_hint_use_negative="""False""", drop_size=10, wildcard="""""", cycle=1, inpaint_model=1, noise_mask_feather=False)
