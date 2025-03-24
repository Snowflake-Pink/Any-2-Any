for TASK in "vanilla" "complex" "creative";
do

    python evaluation_llm.py \
        --inference_engine_name declarative \
        --json_path ./dataset/query/meta_{$TASK}_with_kn.json \
        --save_path ./checkpoint/comfybench/$TASK

done