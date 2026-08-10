for first_layer in 'Spectral' 'Sign' 'ColNorm'
do
    for n_layer in 2 4 8 16 
    do
        for lr in 1 0.5 0.25 0.125 0.0625 0.03125 0.015625 0.0078125 0.00390625 
        do
            for seed in 1 2 3
            do
                head_size=64
                width=256
                n_heads=$((width / head_size))
                out_dir="mup_examples/mutransfer_lr_shakespeare_char/scion/out_depthfactor_2L/hptransferdepth_2L_alpha_skipconnection_first_layer${first_layer}_width${width}_depth${n_layer}_seed${seed}_lr${lr}"
                python train.py \
                    --out_dir=$out_dir \
                    --eval_on_end=False \
                    --eval_iters=200 \
                    --skip_val_loss=False \
                    --eval_only=False \
                    --log_interval=1 \
                    --always_save_checkpoint=False \
                    --never_save_checkpoint=True \
                    --init_from='scratch' \
                    --wandb_log=False \
                    --csv_log=True \
                    --dataset='shakespeare_char' \
                    --gradient_accumulation_steps=1 \
                    --batch_size=32 \
                    --block_size=1024 \
                    --n_layer=$n_layer \
                    --n_head=$n_heads \
                    --n_embd=$width \
                    --dropout=0.0 \
                    --bias=False \
                    --init_std=0.02 \
                    --learning_rate=$lr \
                    --max_iters=122 \
                    --beta1=0.9 \
                    --grad_clip=0.0 \
                    --warmup_iters=0 \
                    --decay_lr=True \
                    --scion_enabled=True \
                    --scion_first_layer=$first_layer \
                    --scion_unconstrained=True \
                    --projected_scion=False \
                    --seed=$seed \
                    --backend='nccl' \
                    --device='cuda' \
                    --dtype='float32' \
                    --compile=False \
                    --mup_enable_coord_check_logging=True
            done
        done
    done
done