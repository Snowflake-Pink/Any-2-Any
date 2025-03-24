from bs4 import BeautifulSoup

from inference_engine.declarative.utils.function import safe_extract_from_soup
from inference_engine.declarative.inference_engine.linker import get_node_knowledge


refiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to design their own workflows.

##introduction

<introduction>
In ComfyUI workflows, the complete pipeline is built in three stages: `add`, `invoke`, and `link`. Each stage has specific responsibilities.

1. Add Node: In this stage, all nodes are instantiated and added to the workflow. This step defines the type of each node and specifies its parameters, such as input images, model checkpoints, or text prompts. The key requirement is that each node must have a unique ID, a node type, and a valid parameter dictionary. Example: `workflow.add_node("cliptextencode_2", "CLIPTextEncode", {{"text": "a beautiful scenery"}})`.

2. Invoke Node: In this stage, the added nodes are executed to perform their computations and produce outputs. The main purpose of invoking nodes is to generate the necessary outputs (e.g., image embeddings, latent variables) that will be used in subsequent stages of the workflow. Each node must be invoked only once. If the same function needs to be reused, a new instance of the node should be created and then invoked. The input variables provided must match the expected input ports of the node to ensure successful execution. Example: workflow.invoke_node(["positive_2"], "cliptextencode_2").

3. Link Node: In this final stage, the nodes are connected to form a complete workflow chain. The outputs of one node are linked to the inputs of another node, ensuring data flows correctly through the pipeline. It is essential to check that the input and output types are compatible; for example, image outputs should be connected to image inputs, and text encodings should be linked to text conditioning inputs. Example: `workflow.connect("latent_3", "vaedecode_8", "samples")`.

These three stages must be executed in order. First, `add_node` initializes all components. Next, `invoke_node` performs the computations. Finally, `link_node` connects the nodes into a coherent workflow. Following these rules ensures the generated workflow is functional and meets the user’s input requirements.

</introduciton>


Now, you are provided with a partially constructed workflow. Your task is to refine the current workflow by fixing any errors and completing it to ensure it functions as expected. You should also check the type of the connection according to the node knowledge.

The user query is as follows:

{query}

## Workspace

The code and description of the current workflow you are working on is presented as follows:

{workspace}

## Nodes knowledge

The corresponding knowledge for the nodes in current workflow is as follows:

{node_knowledge}

## Reference

The code and description of the example workflow you are referring to are presented as follows:

{reference}

## Refinement

However, an error occurred when running your code. This may be caused by missing nodes, incorrect parameter values, or incorrect connections between nodes. The detailed error message is presented as follows:

{refinement}

## Refinement Examples

Here are some examples of how to fix some common errors of invalid code. Each example has the relevant infomation about the codes.
You should learn from the analysis and solution of each example and try to fix the existing error.

<example1>
# Add Node
workflow.add_node("loadimage_3", "LoadImage", {{"image": 'president_trump.jpg'}})
workflow.add_node("ksampler_1", "KSampler", {{"seed": 249584040731174, "control_after_generate": 'randomize', "steps": 20, "cfg": 7, "sampler_name": 'dpmpp_2m', "scheduler": 'karras', "denoise": 1}})

# Invoke Node
workflow.invoke_node(["image_3", "mask_3"], "loadimage_3")
workflow.invoke_node(["latent_1"], "ksampler_1")

# Link Node
workflow.connect("image_3", "ksampler_1", "latent_image")

Error message:
Error parsing code to workflow: workflow.connect("image_3", "ksampler_1", "latent_image"): Type mismatch between IMAGE and LATENT

Analysis and Solution:
Analysis: From the error message we can know that the input type is wrong. In details, 'ksampler_1' requires 'latent_image' of type LATENT but was fed in 'image_3' whose type is IMAGE.
Solution 1: Find the previous node knowledge and check if the existing nodes could provide a latent_image of type LATENT. If yes, use this variable to replace 'image_3'.
Solution 2: Search for required nodes from the reference to provide latent_image of type LATENT to 'ksampler_1'. For example, EmptyLatentImage node was retrieved in this case.

Here are the relevant refined code:
# Add Node
workflow.add_node("loadimage_3", "LoadImage", {{"image": 'president_trump.jpg'}})
workflow.add_node("ksampler_1", "KSampler", {{"seed": 249584040731174, "control_after_generate": 'randomize', "steps": 20, "cfg": 7, "sampler_name": 'dpmpp_2m', "scheduler": 'karras', "denoise": 1}})
workflow.add_node("emptylatentimage_5", "EmptyLatentImage", {{"width": 512, "height": 512, "batch_size": 1}})

# Invoke Node
workflow.invoke_node(["latentimage_5"], "emptylatentimage_5")
workflow.invoke_node(["image_3", "mask_3"], "loadimage_3")
workflow.invoke_node(["latent_1"], "ksampler_1")

# Link Node
workflow.connect("latentimage_5", "ksampler_1", "latent_image")
</example1>

<example2>
# Add Node
workflow.add_node("cliptextencode_9", "CLIPTextEncode", {{"text": '(realistic, high-quality:1.2)'}})
workflow.add_node("cliptextencode_10", "CLIPTextEncode", {{"text": '(low-quality, blurred:1.2)'}})
workflow.add_node("svd_img2vid_conditioning_3", "SVD_img2vid_Conditioning", {{"width": 1024, "height": 576, "video_frames": 24, "motion_bucket_id": 100, "fps": 6, "augmentation_level": 0}})
workflow.add_node("ksampleradvanced_7", "KSamplerAdvanced", {{"add_noise": 'enable', "noise_seed": 49770757027309, "control_after_generate": 'fixed', "steps": 20, "cfg": 2.52, "sampler_name": 'euler', "scheduler": 'ddim_uniform', "start_at_step": 0, "end_at_step": 10000, "return_with_leftover_noise": 'disable'}})
workflow.add_node("imageonlycheckpointloader_15", "ImageOnlyCheckpointLoader", {{"ckpt_name": 'svd.safetensors'}})

# Invoke Node
workflow.invoke_node(["positive_cond_9"], "cliptextencode_9")
workflow.invoke_node(["negative_cond_10"], "cliptextencode_10")
workflow.invoke_node(["positive_cond_3", "negative_cond_3", "latent_3"], "svd_img2vid_conditioning_3")
workflow.invoke_node(["model_15", "clip_vision_15", "vae_15"], "imageonlycheckpointloader_15")
workflow.invoke_node(["latent_7"], "ksampleradvanced_7")

# Link Node
workflow.connect("positive_cond_3", "ksampleradvanced_7", "positive")
workflow.connect("negative_cond_3", "ksampleradvanced_7", "negative")
workflow.connect("latent_3", "ksampleradvanced_7", "latent_image")
workflow.connect("clip_vision_15", "cliptextencode_9", "clip")

Error message:
Error parsing code to workflow: Code line 'workflow.connect("clip_vision_2", "cliptextencode_15", "clip")': Type mismatch between CLIP_VISION and CLIP.

Analysis and Solution:
Analysis: svd_img2vid_conditioning will provide positive_cond and negative_cond, which is enough for the ksampler or ksampleradvanced so do not use extra textencode node. From the error message we can also know that the input type is wrong. In details, 'cliptextencode' requires 'clip' of type CLIP but was fed in 'clip_vision_2' whose type is CLIP_VISION. Besides the two nodes 'cliptextencode_9' and 'cliptextencode_10' are useless because their output won't link to other nodes.
Solution 1: Remove these two useless nodes and their links.
Solution 2: Search for required nodes from the reference to provide clip of type CLIP.

Here are the relevant refined code that apply solution 1:
# Add Node
workflow.add_node("svd_img2vid_conditioning_3", "SVD_img2vid_Conditioning", {{"width": 1024, "height": 576, "video_frames": 24, "motion_bucket_id": 100, "fps": 6, "augmentation_level": 0}})
workflow.add_node("ksampleradvanced_7", "KSamplerAdvanced", {{"add_noise": 'enable', "noise_seed": 49770757027309, "control_after_generate": 'fixed', "steps": 20, "cfg": 2.52, "sampler_name": 'euler', "scheduler": 'ddim_uniform', "start_at_step": 0, "end_at_step": 10000, "return_with_leftover_noise": 'disable'}})

# Invoke Node
workflow.invoke_node(["positive_cond_3", "negative_cond_3", "latent_3"], "svd_img2vid_conditioning_3")
workflow.invoke_node(["latent_7"], "ksampleradvanced_7")

# Link Node
workflow.connect("positive_cond_3", "ksampleradvanced_7", "positive")
workflow.connect("negative_cond_3", "ksampleradvanced_7", "negative")
workflow.connect("latent_3", "ksampleradvanced_7", "latent_image")
</example2>

<exmaple3>
The previous node knowledge:
Node - `unCLIPConditioning`: This node is designed to integrate CLIP vision outputs into the conditioning process, adjusting the influence of these outputs based on specified strength and noise augmentation parameters. It enriches the conditioning with visual context, enhancing the generation process.
    - Parameters:
        - `strength`: Determines the intensity of the CLIP vision output's influence on the conditioning. Type should be `FLOAT`.
        - `noise_augmentation`: Specifies the level of noise augmentation to apply to the CLIP vision output before integrating it into the conditioning. Type should be `FLOAT`.
    - Inputs:
        - `conditioning`: The base conditioning data to which the CLIP vision outputs are to be added, serving as the foundation for further modifications. Type should be `CONDITIONING`.
        - `clip_vision_output`: The output from a CLIP vision model, providing visual context that is integrated into the conditioning. Type should be `CLIP_VISION_OUTPUT`.
    - Outputs:
        - `conditioning`: The enriched conditioning data, now containing integrated CLIP vision outputs with applied strength and noise augmentation. Type should be `CONDITIONING`.

# Add Node
workflow.add_node("unclipconditioning_19", "unCLIPConditioning", {{"strength": 0.5, "noise_augmentation": 0.4000000000000002}})
workflow.add_node("unclipconditioning_37", "unCLIPConditioning", {{"strength": 0.5, "noise_augmentation": 0.4000000000000002}})

# Invoke Node
workflow.invoke_node(["clip_vision_output"], "unclipconditioning_19")
workflow.invoke_node(["clip_vision_output_36"], "unclipconditioning_37")

# Link Node
workflow.connect("clip_vision_output", "unclipconditioning_19", "clip_vision_output")
workflow.connect("clip_vision_output_36", "unclipconditioning_37", "clip_vision_output")

Analysis and Solution:
Analysis: 1. From the node knowlege we can know that the output type of unCLIPConditioning node should be conditioning. Here the invoke part wrongly invoke the input type instead of the output type. 
2. Even worse, it connect to itself after invoke.

Solution: Here are the relevant refined code :
# Invoke Node
workflow.invoke_node(["conditioning_1"], "unclipconditioning_19")
workflow.invoke_node(["conditioning_36"], "unclipconditioning_37")

and unclipconditioning should be connected to previous clip vision's output and conditioning instead.
</example3>

<example4>
# Add Node
workflow.add_node("imageonlycheckpointloader_15", "ImageOnlyCheckpointLoader", {{"ckpt_name": 'stable_zero123.ckpt'}})

# Invoke Node
workflow.invoke_node(["clip_vision_15", "vae_15"], "imageonlycheckpointloader_15")

# Link Node
workflow.connect("clip_vision_15", "stablezero123_conditioning_26", "clip_vision")

Error message:
Error parsing code to workflow: Code line "workflow.connect("clip_vision_15", "stablezero123_conditioning_26", "clip_vision")": Type mismatch between CLIP_VISION and MODEL

Analysis and Solution:
Analysis: From the node knowledge, ImageOnlyCheckpointLoader has three output in order but in your Invoke part you only invoke last two, which makes mistakes in the type of your invocation. 
Solution: You should always invoke all the output in order even if you do not use one of them.

Here are the relevant refined code :
# Invoke Node
workflow.invoke_node(["model_15", "clip_vision_15", "vae_15"], "imageonlycheckpointloader_15")
</example4>

<example5>
Node - `SaveAnimatedWEBP`: The SaveAnimatedWEBP node description.
    - Parameters:
        - `filename_prefix`: Type should be `STRING`.
        - `fps`: Type should be `FLOAT`.
        - `lossless`: Type should be `BOOLEAN`.
        - `quality`: Type should be `INT`.
        - `method`: Type should be `['default', 'fastest', 'slowest']`.
    - Inputs:
        - `images`: Type should be `IMAGE`.
    - Outputs:
    
# Add Node
workflow.add_node("saveanimatedwebp_11", "SaveAnimatedWEBP", {{"filename_prefix": 'ComfyUI', "fps": 10.0, "lossless": False, "quality": 85, "method": 'default'}})

# Invoke Node
workflow.invoke_node(["image_4"], "saveanimatedwebp_11")

# Link Node
workflow.connect("image_4", "saveanimatedwebp_11", "images")

Error message: 
Error during refinement: Output with index 0 not found

Analysis and Solution:
Anallysis: From the node knowledge, node SaveAnimatedWEBP has no output. In your Invoke part, you wrongly invoke it with an image output and even link this non-existent output to itself in Link part.
Solution: Remove the invoke of saveanimatedwebp_11 in Invoke part and connect images input from other source to the saveanimatedwebp_11 in Link part.
</example5>

<example6>
# Add Node
workflow.add_node("svd_img2vid_conditioning_12", "SVD_img2vid_Conditioning", {{"width": 1024, "height": 576, "video_frames": 14, "motion_bucket_id": 127, "fps": 6, "augmentation_level": 0}})
workflow.add_node("loadimage_23", "LoadImage", {{"image": 'mountains.png'}})
workflow.add_node("imageonlycheckpointloader_15", "ImageOnlyCheckpointLoader", {{"ckpt_name": 'svd.safetensors'}})

# Invoke Node
workflow.invoke_node(["model_15", "clip_vision_15", "vae_15"], "imageonlycheckpointloader_15")
workflow.invoke_node(["image_23", "mask_23"], "loadimage_23")

# Link Node
workflow.connect("image_23", "svd_img2vid_conditioning_12", "init_image")
workflow.connect("vae_15", "svd_img2vid_conditioning_12", "vae")

Error message: 
Some nodes' input are missing! Check the link and the knowledge carefully!

Analysis and Solution:
Analysis: From the node knowledge, we can know that SVD_img2vid_Conditioning require init_image, vae and clip_vision as input. In Link part here, you only link the init_image and vae. The required input clip_vison is missing.
Solution: Add the link to svd_img2vid_conditioning:
workflow.connect("clip_vision_15", "svd_img2vid_conditioning_12", "clip_vision")
</example6>

<example7>
Error message: 
ERROR: Error during refinement: Code line "workflow.connect("latent_22", "svd_img2vid_conditioning_12", "init_image")": Type mismatch between IMAGE and LATENT
Analysis and Solution:
Analysis: From the error message we can know that the input type is wrong. In details, 'svd_img2vid_conditioning_12' requires 'init_image' of type IMAGE but was fed in 'latent_22' whose type is LATENT. you should use the image from the vaedecode node instead of the latent.
</example7>

## Fix the problem
First, try to explain why this error occurred.
Your explanation should be enclosed with "<explanation>" tag. For example: <explanation> The error occurred because the input parameter of node_1 is missing. </explanation>.

After that, correct the error and provide your Python code of that good workflow as output again. Your code must meet the following rules:

1. Each code line should either instantiate a node or invoke a node. You should not invoke a node before it is instantiated.
2. Each instantiated node should be invoked only once. You should instantiate another node if you need to reuse the same function.
3. Avoid reusing the same variable name. For example: "workflow.add_node(value_1 , node_1, value_1)" is not allowed because the output "value_1" overrides the input "value_1".
5. Ensure that the provided key nodes are included and correctly linked within the workflow.
6. Make sure the final workflow is complete, with no missing parameters or unconnected nodes.
7. Please note that the Ksampler or Ksampleradvance (if exists) requires negative input, so make sure to have an additional negative input, in addition to the one used for the positive input.
8. Please note that the SVD_img2vid_Conditioning or StableZero123_Conditioning for image to video generation will directly provide the positive condition and the negative condition for Ksampler, so make sure there is no extra textencode node linking to the ksampler. (other tasks like text to video may not suitable)
9. Please note that the ImageOnlyCheckpointLoader for image to video generation will provide model and vae which is enough for further steps, so make sure there is no extra node like CheckpointLoaderSimple that provide extra model and vae. (other tasks like text to video may not suitable)
10. When invoking a node, you must extract all of its outputs, not just the ones currently needed. This ensures that the full range of outputs is available for potential use in later stages of the workflow.
11. Make sure that the nodes in invoke or link period are already defined.
12. Make sure that each node has the inputs it requires reference to the node knowledge.
13. Do not link more than one output to a same port.
14. When invoke nodes, make sure that there will not be same variable name.
15. When invoke and link the nodes, make sure the capitalization and spelling of node names are consistent with the ones in addnode part.
16. Check carefully the type of the input and output of the nodes, make sure they are compatible. And check carefully the connect part according to the node knowledge. Don't leave blank port in invocation like workflow.invoke_node([], "saveimage_9"), instead, do workflow.invoke_node(["result_9"], "saveimage_9") as reference
17. DO NOT SET VIDEO FRAMES MORE THAN 10, IT WILL CAUSE TIMEOUT ERROR.
18. THE LATENT SAMPLES FROM CogVideoImageEncode MUST CONNECT TO THE `image_cond_latents` PORT OF CogVideoSampler INSTEAD OF `samples` PORT
19. If your task is about loading a video and do something on it, i.e. modality is video-to-video, you can just use `VHS_LoadVideo` node to load this video and treat each frame's processing as an batched image-to-image task and finnaly use SaveAnimatedWEBP to save the images to video. You should not use CogVideoSampler node to process V2V tasks. (This means you can just replace the `load image` with `load video` and `save image` with `save video` nodes to adapt functions from image-to-image task to video-to-video task)

Your code should be enclosed with "<code>" tag. For example: <code> workflow.add_node() </code>.

Finally, provide a brief description of the updated workflow and the expected effects as in the example.
Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now, provide your explanation, code, and description with the required format.
'''


def get_refiner_inference_engine_prompt(query: str, linked_code: str, descript: str, error_message: str, reference: str):
    query_content = query

    node_knowledge_content = get_node_knowledge(linked_code)
      
    workspace_content = f'<code>\n{linked_code}\n</code>'
    workspace_content += f'<description>\n{descript}\n</description>'
    refinement_content = error_message.replace("HTTP Error 400: Bad Request", "Some nodes' input are missing! Check all the nodes' invocation and linking with the node knowledge carefully! Then add what's msiing in the correct place")

    prompt_text = refiner_prompt.format(
        node_knowledge = node_knowledge_content,
        workspace=workspace_content,
        refinement=refinement_content,
        query = query_content,
        reference = reference
    )
    return prompt_text


def parse_refiner_inference_engine_response(response: str):
    soup = BeautifulSoup(response, 'html.parser')
    explanation = safe_extract_from_soup(soup, 'explanation')
    code = safe_extract_from_soup(soup, 'code')
    description = safe_extract_from_soup(soup, 'description')
    return explanation, code, description
