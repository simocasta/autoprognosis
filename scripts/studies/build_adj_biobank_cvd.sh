python ./scripts/build_demonstrator.py \
    --name "AutoPrognosis: UK Biobank CVD study" \
    --model_path=./workspace/biobank_cvd/model.p \
    --dataset_path=./workspace/biobank_cvd/biobank_cvd.csv \
    --time_column=time_to_event \
    --target_column=event \
    --horizons="365, 730, 1095, 1460, 1825, 2190, 2555, 2920, 3285, 3650, 4015, 4380" \
    --task_type=risk_estimation \
    --explainers="kernel_shap" \
    --extras=biobank_cvd \
    --auth=True
