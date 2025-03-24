# create nodes by instantiation
LeftBracketcomfy3dRightBracketSpacesetSpacediffusersSpacepipelineSpacestateSpacedict_4 = LeftBracketComfy3DRightBracketSpaceSetSpaceDiffusersSpacePipelineSpaceStateSpaceDict(repo_id="""TencentARC/InstantMesh""", model_name="""diffusion_pytorch_model.bin""")
invertmask_7 = InvertMask()
LeftBracketcomfy3dRightBracketSpacesetSpacediffusersSpacepipelineSpacescheduler_3 = LeftBracketComfy3DRightBracketSpaceSetSpaceDiffusersSpacePipelineSpaceScheduler(diffusers_scheduler_name="""EulerAncestralDiscreteScheduler""")
LeftBracketcomfy3dRightBracketSpaceloadSpacediffusersSpacepipeline_1 = LeftBracketComfy3DRightBracketSpaceLoadSpaceDiffusersSpacePipeline(diffusers_pipeline_name="""Zero123PlusPipeline""", repo_id="""sudo-ai/zero123plus-v1.2""", custom_pipeline="""""", force_download=False, checkpoint_sub_dir="""""")
LeftBracketcomfy3dRightBracketSpacezero123plusSpacediffusionSpacemodel_5 = LeftBracketComfy3DRightBracketSpaceZero123PlusSpaceDiffusionSpaceModel(seed=42, guidance_scale="""fixed""", num_inference_steps=4)
