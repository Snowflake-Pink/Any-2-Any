# create nodes by instantiation
loadimage_33 = LoadImage(image="""cat_stand.jpg""")
vaeencodeforinpaint_38 = VAEEncodeForInpaint(grow_mask_by=0)
cliptextencode_59 = CLIPTextEncode(text="""text, watermark""")
cliptextencode_60 = CLIPTextEncode(text="""a dog""")
groundingdinosamsegmentSpaceLeftParensegmentSpaceanythingRightParen_101 = GroundingDinoSAMSegmentSpaceLeftParensegmentSpaceanythingRightParen(prompt="""cat""", threshold=0.2)
groundingdinomodelloaderSpaceLeftParensegmentSpaceanythingRightParen_102 = GroundingDinoModelLoaderSpaceLeftParensegmentSpaceanythingRightParen(model_name="""GroundingDINO_SwinT_OGC (694MB)""")
