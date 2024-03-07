bash_file_name=$(basename $0)
export CUDA_VISIBLE_DEVICES=1
# for dataset in "cifar100"
# do
#       for seed in 0 1 2 3 4
#       do
#             for tta_method in "GTTA" "RMT" "Source" "BN" "Tent" "SHOT" "SAR" "CoTTA" "RoTTA" "LAW" "PRDCL"
#             do
#             python CTTA.py \
#                   -acfg configs/adapter/${dataset}/${tta_method}.yaml \
#                   -dcfg configs/dataset/${dataset}.yaml \
#                   -ocfg configs/order/${dataset}/0.yaml \
#                   GPU 0 \
#                   SEED $seed \
#                   TEST.BATCH_SIZE 64 \
#                   bash_file_name $bash_file_name \
#                   CORRUPTION.SEVERITY '[5]'
#             done
#       done
# done

for dataset in "cifar100" "cifar10" "imagenet"
do
      for seed in 0
      do
            for tta_method in "PRDCL" "RMT" "LAW" "SAR"
            do
            python CTTA.py \
                  -acfg configs/adapter/${dataset}/${tta_method}.yaml \
                  -dcfg configs/dataset/${dataset}.yaml \
                  -ocfg configs/order/${dataset}/0.yaml \
                  GPU 0 \
                  OUTPUT_DIR log/${tta_method}/${dataset} \
                  SEED $seed \
                  TEST.BATCH_SIZE 64 \
                  bash_file_name $bash_file_name \
                  CORRUPTION.SEVERITY '[5]'
            done
      done
done