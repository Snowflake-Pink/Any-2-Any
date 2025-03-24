# create nodes by instantiation
loadimage_33 = LoadImage(image="""bedroom.jpg""")
lamainpaint_84 = LaMaInpaint(device_mode="""Prefer GPU""")
groundingdinosamsegmentSpaceLeftParensegmentSpaceanythingRightParen_101 = GroundingDinoSAMSegmentSpaceLeftParensegmentSpaceanythingRightParen(prompt="""chair""", threshold=0.2)
groundingdinomodelloaderSpaceLeftParensegmentSpaceanythingRightParen_102 = GroundingDinoModelLoaderSpaceLeftParensegmentSpaceanythingRightParen(model_name="""GroundingDINO_SwinT_OGC (694MB)""")
sammodelloaderSpaceLeftParensegmentSpaceanythingRightParen_103 = SAMModelLoaderSpaceLeftParensegmentSpaceanythingRightParen(model_name="""sam_vit_l (1.25GB)""")
saveimage_113 = SaveImage(filename_prefix="""ComfyUI""")
growmask_114 = GrowMask(expand=45, tapered_corners=True)

# link nodes by invocation
image_33, mask_33 = loadimage_33()
grounding_dino_model_102 = groundingdinomodelloaderSpaceLeftParensegmentSpaceanythingRightParen_102()
sam_model_103 = sammodelloaderSpaceLeftParensegmentSpaceanythingRightParen_103()
image_101, mask_101 = groundingdinosamsegmentSpaceLeftParensegmentSpaceanythingRightParen_101(sam_model=sam_model_103, grounding_dino_model=grounding_dino_model_102, image=image_33)
mask_114 = growmask_114(mask=mask_101)
image_84 = lamainpaint_84(image=image_33, mask=mask_114)
result_113 = saveimage_113(images=image_84)
