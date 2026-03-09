PYTHONWARNINGS=ignore::UserWarning \
python policy_lab/setup_policy_server.py \
    --port 10001 \
    --config_path policy_lab/replay_policy/deploy.yml \
    --overrides \
    --task_name demo_task \
    --policy_name replay_policy \
    --data_path "data/pick_block2/x-one/2.hdf5" \