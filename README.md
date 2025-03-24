<h1 align="center">Symbolic Representation for Any-to-Any Generative Tasks</h1>
<p align="center">

## 🛠️ Configuration

First, create a new conda environment and install the dependencies:

```bash
conda create -n anytoany python=3.12
conda activate anytoany
pip install -r requirements.txt
```

Then, modify the `config.yaml` file to set HTTP proxy address, OpenAI API key, and ComfyUI server address. Before executing the workflows, **you should download our well prepared ComfyUI pack from [https://huggingface.co/JiaqiChen/Any-to-Any-ComfyUI-pack/tree/main](https://huggingface.co/JiaqiChen/Any-to-Any-ComfyUI-pack/tree/main)** or manually install nodes and download models from `nodes_and_models.txt` (not recommended because this could easily get stuck on different versions of custom nodes)

You should also install the environment we provide in `comfyui_env.yml` for ComfyUI setup (notice that your CUDA toolkit version must be 11.8 to run the ComfyUI-3D-pack)

Finally, move the images from `./resources` to `ComfyUI/input`
 to make sure you have all the input images we used in experiment.

## 🚀 Execution

Run the following command to execute the our pipeline directly with `declarative` style:

```bash
bash scripts/easy_run.sh
```

or customize your request by:

```bash
python main.py \
    --query_text "Generate a high-resolution, cinematic image of an anthropomorphic fox in a sci-fi spaceship, wearing a spacesuit, with dramatic lighting and detailed features. The style should be realistic, high quality" \
    --inference_engine_name declarative \
    --save_path ./checkpoint/easy_run
```

`inference_engine_name` could be  `declarative`, `dataflow` or `pseudo_natural` for the three code types mentioned in our paper.


The log file together with the generated workflow will be saved in the specified path. If your ComfyUI server is working properly, the workflow will be executed automatically, and the result will also be saved in the specified path.

To try our multi-task subset, run 

```bash
bash scripts/run_multitask_subset.sh
```
This will run 12 tasks including: image inpaint, image merge, image outpaint, image view inference, image with merged models, image to mesh, image to multiview image, image to video, text to audio, text to image, text to mesh and text to video.
## 🔎 Reproduction

### Multi-task Set

Run the following commands to reproduce the multitask experiments in the paper:

```bash
bash scripts/run_multitask.sh
```

and then evaluate by

```bash
bash scripts/eval_multitask.sh
```

### ComfyBench

**Switch reference**: 

> set `use_comfybench_workflow: true` in `config.yaml`

Run the following commands to reproduce the comfybench experiments in the paper:

```bash
bash scripts/run_comfybench.sh
```

and then evaluate by

```bash
bash scripts/eval_comfybench.sh
```

Make sure that you have set your OpenAI API key in the `config.yaml` file and installed all the required packages and custom nodes like ComfyUI-3D-pack to reproduce our experiment.

## 🤖 Customization

In this section, we provide methods for customizing nodes and references, so that users can explore more abilities of our inference engine beyond the 12 tasks we have set and the tasks set by Comfybench.

### Add your own custom nodes template
modify the `SAVEPATH` in `./tools/generate_custom_node_template.py` and move it to the root dir of your ComfyUI and add the following lines in `ComfyUI/nodes.py` : from

```python
if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
    for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
        if name not in ignore:
            NODE_CLASS_MAPPINGS[name] = node_cls
```

to

```python
from generate_custom_node_template import generate_custom_template
if hasattr(module, "NODE_CLASS_MAPPINGS") and getattr(module, "NODE_CLASS_MAPPINGS") is not None:
    for name, node_cls in module.NODE_CLASS_MAPPINGS.items():
        if name not in ignore:
            NODE_CLASS_MAPPINGS[name] = node_cls
            try:
                template = generate_custom_template(node_cls, name)
            except Exception as e:
                print(e)
```

we also provide a modified version of `nodes.py` in `./tools` so you don't need to modified the above lines if you replace the file. ( Remeber to backup raw file in case of this not work due to differnt version or other issues)

Now start ComfyUI server and it will automatically generate custom node templates to the `SAVEPATH`. 

### Add your own workflow to reference

* First, modify the path in `tools/raw_to_code_and_md.py` and run. This will generate the `.py` code and `.md` file of your workflow.

* Then, move your workflow to `dataset/workflow/raw`, the `.py` code to `dataset/workflow/code`, the `.md` file to `dataset/workflow/md`.

* Next, add the description of your workflow in `dataset/workflow/desc`.

* Finally, add the path in `dataset/workflow/meta.json` and remove saved database `dataset/workflow/db` folder.

Following the steps above and then inference engine will retrieve your customize reference.