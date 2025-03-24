for TASK in "vanilla" "complex" "creative";
do
    python inference.py \
            --inference_engine_name declarative \
            --json_path ./dataset/query/meta_{$TASK}_with_kn.json \
            --num_runs 1 \
            --save_path ./checkpoint/comfybench/$TASK
done