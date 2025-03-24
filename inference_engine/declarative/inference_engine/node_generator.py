from bs4 import BeautifulSoup

from inference_engine.declarative.utils.function import safe_extract_from_soup


combiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to instantiate their own workflows nodes.

##introduction

<introduction>
In ComfyUI workflows, the complete pipeline is built in three stages: `add`, `invoke`, and `link`. Each stage has specific responsibilities.

1. Add Node: In this stage, all nodes are instantiated and added to the workflow. This step defines the type of each node and specifies its parameters, such as input images, model checkpoints, or text prompts. The key requirement is that each node must have a unique ID, a node type, and a valid parameter dictionary. Example: `workflow.add_node("cliptextencode_2", "CLIPTextEncode", {{"text": "a beautiful scenery"}})`.

2. Invoke Node: In this stage, the added nodes are executed to perform their computations and produce outputs. The main purpose of invoking nodes is to generate the necessary outputs (e.g., image embeddings, latent variables) that will be used in subsequent stages of the workflow. Each node must be invoked only once. If the same function needs to be reused, a new instance of the node should be created and then invoked. The input variables provided must match the expected input ports of the node to ensure successful execution. Example: workflow.invoke_node(["positive_2"], "cliptextencode_2").

3. Link Node: In this final stage, the nodes are connected to form a complete workflow chain. The outputs of one node are linked to the inputs of another node, ensuring data flows correctly through the pipeline. It is essential to check that the input and output types are compatible; for example, image outputs should be connected to image inputs, and text encodings should be linked to text conditioning inputs. Example: `workflow.connect("latent_3", "vaedecode_8", "samples")`.

These three stages must be executed in order. First, `add_node` initializes all components. Next, `invoke_node` performs the computations. Finally, `link_node` connects the nodes into a coherent workflow. Following these rules ensures the generated workflow is functional and meets the user’s input requirements.

</introduciton>

Now you are required to create a ComfyUI workflow to finish the following task:

The user query is as follows:

{query}

The key points behind the requirements and the expected paradigm of the workflow are analyzed as follows:

{analysis}

## Reference

The code and description of the example workflow you are referring to are presented as follows:

{reference}

## Key Nodes
The user has provided the following key nodes that *must be included* in the workflow:

{key_nodes}

The corresponding knowledge for the key nodes is as follows:

{key_node_knowledge}

## Workspace

The code of the current nodes instantiation you are working on are presented as follows:

{code}

## Combination and Adaptation

Following the current progress, the step-by-step plan is outlined as follows:

1. Select Key Nodes: Select all nodes in the key nodes.
2. Analyze Knowledge and Queries: Review the information associated with the key nodes and evaluate the query.
3. Construct the Workflow Nodes: Base on the key nodes, choose the rest nodes that can assist in building the workflow from the workspace, followed by selecting nodes from the reference.
4. Adapt to New Inputs: Integrate new values (e.g., input text) into the selected nodes to fulfill the requirements but *do not* modify the model_name or ckpt_name. For example, change 'CheckpointLoaderSimple(ckpt_name="""dreamshaper_8.safetensors""")' to 'CheckpointLoaderSimple(ckpt_name="""model.safetensors""")' is not allowed because this modify the ckpt_name.

In other words, you should merge the nodes instantiation in example workflow with the current nodes instantiation code according to your plan, so that their functions can be combined.

First, you should provide your Python code to initiate all the nodes needed for the updated workflow. Your code must meet the following rules:
Please note that the K-sampler requires negative input, so make sure to have an additional CONDITIONING type variable from invoke specifically for the negative input, in addition to the one used for the positive input.

## Other Requirements

1. Each code line should either instantiate a node or invoke a node. You should not invoke a node before it is instantiated. When you add node, please make sure you initiate correct parameters.
2. Stick with reference the best way you can, and **never** delete keynodes given by the user.
3. Avoid reusing the same variable name. For example: "workflow.add_node(value_1 , node_1, value_1)" is not allowed because the output "value_1" overrides the input "value_1".
4. Your code should only contain the complete 'add node' part of the workflow, but not include invoke/links.
5. Make sure all the key nodes are included and necessary extra nodes for the query are added, but don't create nodes yourself(you need to have a reference for every node you add).
6. Before your output your code, check the validity of all the parameters in node instantiation. You should fill the nodes in valid and sound texts/conditions/models/parameters that could work without any manual adjustment.
7. Please note that the SVD_img2vid_Conditioning or StableZero123_Conditioning for image to video generation will directly provide the positive condition and the negative condition for Ksampler, so make sure there is no extra textencode node provided for the ksampler. (other tasks like text to video may not suitable)
8. Please note that the ImageOnlyCheckpointLoader for image to video generation will provide model and vae which is enough for further steps, so make sure there is no extra node like CheckpointLoaderSimple that provide extra model and vae. (other tasks like text to video may not suitable)
9. Please follow the standard format of node adding, the name shoud consists of node name and '_' followed by ** a number **, like this: workflow.add_node("vaedecode_23", "VAEDecode", {{}}). Check the nodes you add follow this format. For example, workflow.add_node("loadimage_beach", "LoadImage", {{"image": 'beach.png'}}) is wrong, while workflow.add_node("loadimage_1", "LoadImage", {{"image": 'beach.png'}}) is correct.
10. A query ask for video generation with no stated input is a text to video query. Note that in text to video task, you always need to include key words like 'significant perspective change' or explicity require large motion in your prompt.
11. DO NOT SET VIDEO FRAMES MORE THAN 10, IT WILL CAUSE TIMEOUT ERROR. VIDEO SAMPLING STEP SHOULD BE LESS THAN 50 AND GREATER THAN 20 TO IMPROVE QUALITY.
12. If you are required to generate a video (including upscale video frames), use SaveAnimatedWEBP as the lastest node to save the video. Also, remember 1.add 'cliploader' node to prepare the model for the cogvideo text encode. 2.add 'downloadandloadcogvideomodel' node to prepare for the model and vae.
13. If you need to load video, use `VHS_LoadVideo` node. For example: workflow.add_node("vhs_loadvideo_1", "VHS_LoadVideo", {{"video": 'male_idol.mp4', "force_rate": 0, "force_size": 'Disabled', "custom_width": 512, "custom_height": 512, "frame_load_cap": 0, "skip_first_frames": 0, "select_every_nth": 1}})
14. If your task is about loading a video and do something on it, i.e. modality is video-to-video, you can just use `VHS_LoadVideo` node to load this video and treat each frame's processing as an batched image-to-image task and finnaly use SaveAnimatedWEBP to save the video. You should not use CogVideoSampler node to process V2V tasks. (This means you can just replace the `load image` with `load video` and `save image` with `save video` nodes to adapt functions from image-to-image task to video-to-video task)
15. Do not set *parameters initial value* (not port named `negative`, just the value) to negative value unless the node needs exactly or explicitly stated in the task description. For example, do not set `CR Overlay Text`'s position to a negative value.
16. When you are adding prompt to remove/replace object, be specific and concise (3-7words)

Your code should be enclosed with "<code>" tag. This is an example for format only: 

<code> 
# Add Node
workflow.add_node("vaeencodeforinpaint_12", "VAEEncodeForInpaint", {{"grow_mask_by": 16}})
workflow.add_node("checkpointloadersimple_4", "CheckpointLoaderSimple", {{"ckpt_name": 'dreamshaper_8.safetensors'}})
workflow.add_node("vaeloader_70", "VAELoader", {{"vae_name": 'vae-ft-mse-840000-ema-pruned.safetensors'}})
workflow.add_node("checkpointloadersimple_25", "CheckpointLoaderSimple", {{"ckpt_name": 'dreamshaper_8Inpainting.safetensors'}})
workflow.add_node("loadimage_78", "LoadImage", {{"image": 'iceberg.jpg'}})
workflow.add_node("cliptextencode_7", "CLIPTextEncode", {{"text": 'illustration, painting, text, watermark, copyright, signature, notes'}})
workflow.add_node("ksampler_21", "KSampler", {{"seed": 1, "control_after_generate": 'fixed', "steps": 20, "cfg": 7, "sampler_name": 'dpmpp_2m', "scheduler": 'karras', "denoise": 1}})
workflow.add_node("imagepadforoutpaint_11", "ImagePadForOutpaint", {{"left": 256, "top": 0, "right": 256, "bottom": 0, "feathering": 0}})
workflow.add_node("cliptextencode_6", "CLIPTextEncode", {{"text": 'an image of iceberg'}})
workflow.add_node("saveimage_79", "SaveImage", {{"filename_prefix": 'ComfyUI'}})
workflow.add_node("vaedecode_23", "VAEDecode", {{}}) 
</code>.

After that, you should provide a brief description of the whole set of node instantiations and their intended roles in the workflow.
Your description should be enclosed with "<description>" tag. For example: <description> Added an upscaling module to enhance image resolution. </description>.

Now, check that you don't miss any nodes in the user's keynodes, and provide your code and description with the required format (specifically, if this is a text to video, remember to add this: workflow.add_node("checkpointloadersimple_16", "CheckpointLoaderSimple", {{"ckpt_name": 'sd_xl_base_1.0.safetensors'}})).
'''

def get_generator_inference_engine_prompt(
    code:str,
    query:str,
    analysis:str,
    keyword:str,
    reference:str,
    key_nodes:str,
    key_node_knowledge:str
):

    prompt_text = combiner_prompt.format(
        code=f'<code>\n{code}\n</code>\n\n',
        query=query,
        analysis=analysis,
        reference=reference,
        key_nodes=f'<code>\n{key_nodes}\n</code>\n\n',
        key_node_knowledge=key_node_knowledge
    )
    return prompt_text

def parse_generator_inference_engine_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    code = safe_extract_from_soup(soup, 'code')
    description = safe_extract_from_soup(soup, 'description')
    return code, description