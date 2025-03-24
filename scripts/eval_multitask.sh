for TASK in "image_inpaint" "image_merge" "image_outpaint" "image_view_inference" "image_with_merged_models" \
            "image2multiview_image" "image2video" "text2audio" "text2image" "text2video" "image2mesh" "text2mesh";
do
    for inference_engine in "declarative" "dataflow" "pseudo_natural";
    do
        python evaluation.py \
            --inference_engine_name $inference_engine \
            --json_path ./workspace/multi_task_set/${TASK}/meta.json \
            --save_path ./checkpoint/multi_task_set/$TASK
    done
done