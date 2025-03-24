# create nodes by instantiation
ksampler_3 = KSampler(seed=52117596413767, control_after_generate="""randomize""", steps=20, cfg=7, sampler_name="""dpmpp_3m_sde_gpu""", scheduler="""sgm_uniform""", denoise=1)
emptylatentimage_5 = EmptyLatentImage(width=768, height=768, batch_size=1)
cliptextencode_6 = CLIPTextEncode(text="""a beautiful photograph of an old European city""")
cliptextencode_7 = CLIPTextEncode(text="""""")
