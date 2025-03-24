# create nodes by instantiation
checkpointloadersimple_4 = CheckpointLoaderSimple(ckpt_name="""majicmixRealistic_v7.safetensors""")
cliptextencode_6 = CLIPTextEncode(text="""1girl, """)
cliptextencode_7 = CLIPTextEncode(text="""lowres,zombie,horror,nsfw, """)
facedetailer_11 = FaceDetailer(guide_size=384, guide_size_for=True, max_size=1024, seed=266448747412199, control_after_generate="""randomize""", steps=20, cfg=4, sampler_name="""euler_ancestral""", scheduler="""normal""", denoise=0.5, feather=5, noise_mask=True, force_inpaint=True, bbox_threshold=0.5, bbox_dilation=10, bbox_crop_factor=3, sam_detection_hint="""center-1""", sam_dilation=0, sam_threshold=0.93, sam_bbox_expansion=0, sam_mask_hint_threshold=0.7000000000000001, sam_mask_hint_use_negative="""False""", drop_size=10, wildcard="""""", cycle=1, inpaint_model=1, noise_mask_feather=False)
saveimage_17 = SaveImage(filename_prefix="""ComfyUI""")
ultralyticsdetectorprovider_18 = UltralyticsDetectorProvider(model_name="""bbox/face_yolov8m.pt""")
samloader_19 = SAMLoader(model_name="""sam_vit_b_01ec64.pth""", device_mode="""AUTO""")
loadimage_26 = LoadImage(image="""woman_portrait.jpg""")

# link nodes by invocation
model_4, clip_4, vae_4 = checkpointloadersimple_4()
conditioning_7 = cliptextencode_7(clip=clip_4)
bbox_detector_18, segm_detector_18 = ultralyticsdetectorprovider_18()
image_26, mask_26 = loadimage_26()
conditioning_6 = cliptextencode_6(clip=clip_4)
sam_model_19 = samloader_19()
image_11, cropped_refined_11, cropped_enhanced_alpha_11, mask_11, detailer_pipe_11, cnet_images_11 = facedetailer_11(image=image_26, model=model_4, clip=clip_4, vae=vae_4, positive=conditioning_6, negative=conditioning_7, bbox_detector=bbox_detector_18, sam_model_opt=sam_model_19, segm_detector_opt=segm_detector_18, detailer_hook=None, scheduler_func_opt=None)
result_17 = saveimage_17(images=image_11)
