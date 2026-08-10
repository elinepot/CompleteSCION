
for n_layer in 2 4 8 16 32
do
    for lr in 0.001953125, 0.0009765625, 0.00048828125, 0.000244140625, 0.0001220703125, 6.103515625e-05
    do
        for seed in 1 2 3
        do
            torchrun --nproc_per_node=4  train_gpt_scion.py \
            --num-iterations=10 \
            --learning-rate=$lr \
            --n-layer=$n_layer \
            --unconstrained \
            --seed=$seed \
            --alpha-skip-connection=1.0
        done
    done
done
