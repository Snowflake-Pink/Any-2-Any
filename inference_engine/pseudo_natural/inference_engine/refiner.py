from bs4 import BeautifulSoup

from inference_engine.pseudo_natural.utils.function import safe_extract_from_soup
from inference_engine.pseudo_natural.inference_engine.linker import get_node_knowledge


refiner_prompt = '''
## Task

ComfyUI uses workflows to create and execute Stable Diffusion pipelines so that users can design their own workflows to generate highly customized artworks.
ComfyUI provides many nodes. Each node represents a module in the pipeline. Users can formulate a workflow into Python code by instantiating nodes and invoking them for execution.
You are an expert in ComfyUI who helps users to design their own workflows.

Now, you are provided with a partially constructed workflow. Your task is to refine the current workflow by fixing any errors and completing it to ensure it functions as expected.

The user query is as follows:

{query}

## Workspace

The code and description of the current workflow you are working on is presented as follows:

{workspace}

## Nodes knowledge

*Note that brackets of nodes are represented by words of their names in codelines (e.g. LeftBracketComfy3DRightBracket_TripoSR represent the node [Comfy3D] TripoSR), so you need to be careful to match them with the corresponding node knowledge.
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
# create nodes by instantiation
loadimage_3 is LoadImage with the parameters of (image is 'president_trump.jpg')
ksampler_1 is KSampler with the parameters of (seed is 249584040731174, control_after_generate is 'randomize', steps is 20, cfg is 7, sampler_name is 'dpmpp_2m', scheduler is 'karras', denoise is 1)

# link nodes by invocation
image_3, mask_3 is loadimage_3()
latent_1 is ksampler_1 with the parameters of (model is model_4, positive is conditioning_16, negative is conditioning_7, latent_image is image_3)

Error message:
Error parsing code to workflow: code linelatent_1 = ksampler_1(model=model_4, positive=conditioning_16, negative=conditioning_7, latent_image=image_3): type mismatch between IMAGE and LATENT

Analysis and Solution:
Analysis: From the error message we can know that the input type is wrong. In details, 'ksampler_1' requires 'latent_image' of type LATENT but was fed in 'image_3' whose type is IMAGE.
Solution 1: Find the previous node knowledge and check if the existing nodes could provide a latent_image of type LATENT. If yes, use this variable to replace 'image_3'.
Solution 2: Search for required nodes from the reference to provide latent_image of type LATENT to 'ksampler_1'. For example, EmptyLatentImage node was retrieved in this case.

Here are the relevant refined code:
# create nodes by instantiation
loadimage_3 is LoadImage with the parameters of (image is 'president_trump.jpg')
ksampler_1 is KSampler with the parameters of (seed is 249584040731174, control_after_generate is 'randomize', steps is 20, cfg is 7, sampler_name is 'dpmpp_2m', scheduler is 'karras', denoise is 1)
emptylatentimage_5 is EmptyLatentImage with the parameters of (width is 512, height is 512, batch_size is 1)

# link nodes by invocation
latentimage_5 is emptylatentimage_5()
image_3, mask_3 is loadimage_3()
latent_1 is ksampler_1 with the parameters of (model is model_4, positive is conditioning_16, negative is conditioning_7, latent_image is latentimage_5)
</example1>

<example2>
# create nodes by instantiation
cliptextencode_9 is CLIPTextEncode with the parameters of (text is '(realistic, high-quality:1.2)')
cliptextencode_10 is CLIPTextEncode with the parameters of (text is '(low-quality, blurred:1.2)')
svd_img2vid_conditioning_3 is SVD_img2vid_Conditioning with the parameters of (width is 1024, height is 576, video_frames is 24, motion_bucket_id is 100, fps is 6, augmentation_level is 0)
ksampleradvanced_7 is KSamplerAdvanced with the parameters of (add_noise is 'enable', noise_seed is 49770757027309, control_after_generate is 'fixed', steps is 20, cfg is 2.52, sampler_name is 'euler', scheduler is 'ddim_uniform', start_at_step is 0, end_at_step is 10000, return_with_leftover_noise is 'disable')

# link nodes by invocation
positive_cond_9 is cliptextencode_9 with the parameters of (clip is clip_vision_2)
negative_cond_10 is cliptextencode_10 with the parameters of (clip is clip_vision_2)
positive_cond_3, negative_cond_3, latent_3 is svd_img2vid_conditioning_3 with the parameters of (clip_vision is clip_vision_2, init_image is image_1, vae is vae_2)
latent_7 is ksampleradvanced_7 with the parameters of (model is model_6, positive is positive_cond_3, negative is negative_cond_3, latent_image is latent_3)

Error message:
Error parsing code to workflow: code line positive_cond_9 is cliptextencode_9 with the parameters of (clip=clip_vision_2): type mismatch between CLIP_VISION and CLIP.

Analysis and Solution:
Analysis: svd_img2vid_conditioning will provide positive_cond and negative_cond, which is enough for the ksampler or ksampleradvanced so do not use extra textencode node. From the error message we can also know that the input type is wrong. In details, 'cliptextencode' requires 'clip' of type CLIP but was fed in 'clip_vision_2' whose type is CLIP_VISION. Besides the two nodes 'cliptextencode_9' and 'cliptextencode_10' are useless because their output won't link to other nodes.
Solution 1: Remove these two useless nodes and their links.
Solution 2: Search for required nodes from the reference to provide clip of type CLIP.

Here are the relevant refined code that apply solution 1:
# create nodes by instantiation
svd_img2vid_conditioning_3 is SVD_img2vid_Conditioning with the parameters of (width is 1024, height is 576, video_frames is 24, motion_bucket_id is 100, fps is 6, augmentation_level is 0)
ksampleradvanced_7 is KSamplerAdvanced with the parameters of (add_noise is 'enable', noise_seed is 49770757027309, control_after_generate is 'fixed', steps is 20, cfg is 2.52, sampler_name is 'euler', scheduler is 'ddim_uniform', start_at_step is 0, end_at_step is 10000, return_with_leftover_noise is 'disable')

# link nodes by invocation
positive_cond_3, negative_cond_3, latent_3 is svd_img2vid_conditioning_3 with the parameters of (clip_vision is clip_vision_2, init_image is image_1, vae is vae_2)
latent_7 is ksampleradvanced_7 with the parameters of (model is model_6, positive is positive_cond_3, negative is negative_cond_3, latent_image is latent_3)
</example2>

<example3>
# create nodes by instantiation
imageonlycheckpointloader_15 is ImageOnlyCheckpointLoader with the parameters of (ckpt_name is 'svd_xt.safetensors')
svd_img2vid_conditioning_12 is SVD_img2vid_Conditioning with the parameters of (width is 1024, height is 576, video_frames is 25, motion_bucket_id is 127, fps is 6, augmentation_level is 0)
videolinearcfgguidance_14 is VideoLinearCFGGuidance with the parameters of (min_cfg is 1)
ksampler_17 is KSampler with the parameters of (seed is 1043315181093700, control_after_generate is 'randomize', steps is 15, cfg is 8, sampler_name is 'uni_pc_bh2', scheduler is 'normal', denoise is 1)
vaedecode_20 is VAEDecode()
ksampler_3 is KSampler with the parameters of (seed is 473009707600340, control_after_generate is 'randomize', steps is 20, cfg is 2.5, sampler_name is 'euler', scheduler is 'karras', denoise is 1)
vaedecode_8 is VAEDecode()
saveanimatedwebp_10 is SaveAnimatedWEBP with the parameters of (filename_prefix is 'ComfyUI', fps is 10, lossless is False, quality is 85, method is 'default')
emptylatentimage_22 is EmptyLatentImage with the parameters of (width is 1024, height is 576, batch_size is 1)

# link nodes by invocation
latent_image_22 is emptylatentimage_22()
model_15, clip_vision_15, vae_15 is imageonlycheckpointloader_15()
positive_conditioning_12, negative_conditioning_12, latent_conditioning_12 is svd_img2vid_conditioning_12 with the parameters of (clip_vision is clip_vision_15, vae is vae_15)
latent_17 is ksampler_17 with the parameters of (model is model_15, positive is positive_conditioning_12, negative is negative_conditioning_12, latent_image is latent_image_22)
image_20 is vaedecode_20 with the parameters of (samples is latent_17, vae is vae_15)
enhanced_model_14 is videolinearcfgguidance_14 with the parameters of (model is model_15)
latent_3 is ksampler_3 with the parameters of (model is enhanced_model_14, positive is positive_conditioning_12, negative is negative_conditioning_12, latent_image is latent_conditioning_12)
image_8 is vaedecode_8 with the parameters of (samples is latent_3, vae is vae_15)
result_10 is saveanimatedwebp_10 with the parameters of (images is image_8)

Error message:
Error executing the workflow: Node svd_img2vid_conditioning_12 Required input is missing:init image.

Analysis and Solution:
Analysis: This code represent a text to video workflow. This workflow can be devided into two parts: text to image and image to video.
From the error message we can know that there should be a input image from the text to image stage for svd_img2vid_conditioning_12 to processing image to video stage.

Solution: 
1. Firstly, check whether the two checkpoint loaders (one is imageonlycheckpointloader for image to video and another is checkpointloadersimple for text to image) are correctly used in two stage. If any one is missing, add it from the reference.
2. Then, check whether the two vae and the two clip exist and whether they correctly connected to two decode node respectively.
3. Next, check the input of svd_img2vid_conditioning_12
4. Finally, make sure the saveanimatedwebp node only receive the images from the image to video stage decode. 

Here are the relevant refined code:
# create nodes by instantiation
videolinearcfgguidance_14 is VideoLinearCFGGuidance with the parameters of (min_cfg is 1)
svd_img2vid_conditioning_12 is SVD_img2vid_Conditioning with the parameters of (width is 1024, height is 576, video_frames is 25, motion_bucket_id is 127, fps is 6, augmentation_level is 0)
ksampler_3 is KSampler with the parameters of (seed is 473009707600340, control_after_generate is 'randomize', steps is 20, cfg is 2.5, sampler_name is 'euler', scheduler is 'karras', denoise is 1)
vaedecode_8 is VAEDecode()
previewimage_21 is PreviewImage()
vaedecode_20 is VAEDecode()
ksampler_17 is KSampler with the parameters of (seed is 1043315181093700, control_after_generate is 'randomize', steps is 15, cfg is 8, sampler_name is 'uni_pc_bh2', scheduler is 'normal', denoise is 1)
emptylatentimage_22 is EmptyLatentImage with the parameters of (width is 1024, height is 576, batch_size is 1)
cliptextencode_19 is CLIPTextEncode with the parameters of (text is 'text, watermark')
imageonlycheckpointloader_15 is ImageOnlyCheckpointLoader with the parameters of (ckpt_name is 'svd_xt.safetensors')
saveanimatedwebp_10 is SaveAnimatedWEBP with the parameters of (filename_prefix is 'ComfyUI', fps is 10, lossless is False, quality is 85, method is 'default')
checkpointloadersimple_16 is CheckpointLoaderSimple with the parameters of (ckpt_name is 'sd_xl_base_1.0.safetensors')
cliptextencode_18 is CLIPTextEncode with the parameters of (text is 'a cup of coffee being poured, but instead of coffee, a miniature galaxy swirls out, with stars and planets floating in the liquid')

# link nodes by invocation
latent_22 is emptylatentimage_22()
model_15, clip_vision_15, vae_15 is imageonlycheckpointloader_15()
model_16, clip_16, vae_16 is checkpointloadersimple_16()
model_14 is videolinearcfgguidance_14 with the parameters of (model is model_15)
conditioning_19 is cliptextencode_19 with the parameters of (clip is clip_16)
conditioning_18 is cliptextencode_18 with the parameters of (clip is clip_16)
latent_17 is ksampler_17 with the parameters of (model is model_16, positive is conditioning_18, negative is conditioning_19, latent_image is latent_22)
image_20 is vaedecode_20 with the parameters of (samples is latent_17, vae is vae_16)
positive_12, negative_12, latent_12 is svd_img2vid_conditioning_12 with the parameters of (clip_vision is clip_vision_15, init_image is image_20, vae is vae_15)
result_21 is previewimage_21 with the parameters of (images is image_20)
latent_3 is ksampler_3 with the parameters of (model is model_14, positive is positive_12, negative is negative_12, latent_image is latent_12)
image_8 is vaedecode_8 with the parameters of (samples is latent_3, vae is vae_15)
result_10 is saveanimatedwebp_10 with the parameters of (images is image_8)
</example3>

<example4>
# create nodes by instantiation
imageonlycheckpointloader_15 is ImageOnlyCheckpointLoader with the parameters of (ckpt_name is stable_zero123.ckpt)

# link nodes by invocation
clip_vision_15, vae_15 is imageonlycheckpointloader_15()

Analysis and Solution:
Analysis: From the node knowledge, ImageOnlyCheckpointLoader has three output in order but in your link part you only output last two, which makes mistakes in type. 
Solution: You should always invoke all the output in order even if you do not use one of them.

Here are the relevant refined code :
model_15, clip_vision_15, vae_15 is imageonlycheckpointloader_15()
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
    
# create nodes by instantiation
saveanimatedwebp_11 is SaveAnimatedWEBP with the parameters of (filename_prefix is ComfyUI, fps is 10.0, lossless is False, quality is 85, method is 'default')

# link nodes by invocation
image_4 is saveanimatedwebp_11 with the parameters of (images is image_4)

Error message: 
Error during refinement: Output with index 0 not found

Analysis and Solution:
Anallysis: From the node knowledge, node SaveAnimatedWEBP has no output. In your link part, you wrongly invoke it with an image output and even link this non-existent output to itself in Link part.
Solution: Remove the invoke of saveanimatedwebp_11 in Invoke part and connect images input from other source to the saveanimatedwebp_11 in Link part.
</example5>

<example6>
# create nodes by instantiation
svd_img2vid_conditioning_12 is SVD_img2vid_Conditioning with the parameters of (width is 1024, height is 576, video_frames is 14, motion_bucket_id is 127, fps is 6, augmentation_level is 0)

# link nodes by invocation
positive_12, negative_12, latent_12 is svd_img2vid_conditioning_12 with the parameters of (init_image is image_23, vae is vae_15)

Error message: 
Some nodes' input are missing! Check the link and the knowledge carefully!

Analysis and Solution:
Analysis: From the node knowledge, we can know that SVD_img2vid_Conditioning require init_image, vae and clip_vision as input. In Link part here, you only link the init_image and vae. The required input clip_vison is missing.
Solution: Add the link to svd_img2vid_conditioning:
positive_12, negative_12, latent_12 is svd_img2vid_conditioning_12 with the parameters of (clip_vison is clip_vision_15, init_image is image_23, vae is vae_15)
</example6>

## Fix the problem
First, try to explain why this error occurred.
Your explanation should be enclosed with "<explanation>" tag. For example: <explanation> The error occurred because the input parameter of node_1 is missing. </explanation>.

After that, correct the error and provide your Python code of that good workflow as output again. Your code must meet the following rules:

1. Each code line should either instantiate a node or invoke a node. You should not invoke a node before it is instantiated.
2. Each instantiated node should be invoked only once. You should instantiate another node if you need to reuse the same function.
3. Avoid reusing the same variable name. For example: "value_1 is node_1 with the parameters of (value_1)" is not allowed because the output "value_1" overrides the input "value_1".
4. Avoid nested calls in a single code line. For example: "output_2 is node_2 with the parameters of (input_1, node_1())" should be separated into "output_1 is node_1() and output_2 is node_2 with the parameters of (input_1, output_1)". Another example is: "output_2 is node_2 with the parameters of (node_1()[0], node_1()[1])" should be separated into "output_1, output_2 is node_1() and output_2 is node_2 with the parameters of (output_1, output_2)". 
5. Ensure that the provided key nodes are included and correctly linked within the workflow.
6. Make sure the final workflow is complete, with no missing parameters or unconnected nodes.
7. Please note that the Ksampler or Ksampleradvance (if exists) requires negative input, so make sure to have an additional negative input, in addition to the one used for the positive input.
8. Please note that the SVD_img2vid_Conditioning for image to video generation will directly provide the positive condition and the negative condition for Ksampler, so make sure there is no extra textencode node linking to the ksampler.
9. Please note that the ImageOnlyCheckpointLoader for image to video generation will provide model and vae which is enough for further steps, so make sure there is no extra node like CheckpointLoaderSimple that provide extra model and vae.
10. When linking the node, you must extract all of its outputs, not just the ones currently needed. This ensures that the full range of outputs is available for potential use in later stages of the workflow.
11. Make sure that the nodes in link period are already defined.
12. Make sure that each node has the inputs it requires reference to the node knowledge.
13. Do not link more than one output to a same port.
14. When linking nodes, make sure that there will not be same variable name.
15. When linking the nodes, make sure the capitalization and spelling of node names are consistent with the ones in addnode part.

Your code should be enclosed with "<code>" tag. For example: <code> output_1 is node_1() </code>.

Finally, provide a brief description of the updated workflow and the expected effects as in the example.
Your description should be enclosed with "<description>" tag. For example: <description> This workflow uses the text-to-image pipeline together with an upscaling module to generate a high-resolution image of a running horse. </description>.

Now, provide your explanation, code, and description with the required format.
'''


def get_refiner_inference_engine_prompt(query: str, linked_code: str, descript: str, error_message: str, reference: str):
    query_content = query

    node_knowledge_content = get_node_knowledge(linked_code)
      
    workspace_content = f'<code>\n{linked_code}\n</code>'
    workspace_content += f'<description>\n{descript}\n</description>'
    refinement_content = error_message

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
