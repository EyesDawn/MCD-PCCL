# --gpu 2 # echo 'export PYTHONPATH = "/workspace/reference/ts2vec"'
# echo 'conda activate mstad' 
# python trainbk.py HAR all_perc --loader HAR --batch-size 256 --gpu 0 #--data_perc "5perc"  

lab_percs=( "5perc") #  "5perc" "10perc" "30perc" "50perc" "75perc"

for lab_perc in "${lab_percs[@]}"
do
python3 trainbk.py Bridge all_perc --loader HAR --batch-size 256 --data_perc $lab_perc  --gpu 6 --eval --model_path ./Civil/show/HAR/all_perc_train/20241211_091248/model.pkl

#     python train_dual.py Bridge civil --loader HAR --batch-size 256 --data_perc 5perc --gpu 2
#     python train_dual.py RoadBank civil --loader HAR --batch-size 256 --data_perc 5perc --gpu 2
    # python trainbk.py Bridge civil_bi --loader HAR --batch-size 256 --data_perc $lab_perc  --gpu 0    
    # python trainbk.py RoadBank civil_bi --loader HAR --batch-size 256 --data_perc 50perc  --gpu 0
#     # python train_dual.py Epilepsy edf_topk15 --loader HAR --batch-size 512 --data_perc 1perc --gpu 2
#     # python train_dual.py HAR edf_topk10 --loader HAR --batch-size 512 --data_perc 2perc --gpu 2
#     # python train_dual.py FordA edf_topk15 --loader UCR --batch-size 64 --data_perc 1perc --gpu 2
#     # python train_dual.py FordB edf_topk15 --loader UCR --batch-size 64 --data_perc 1perc --gpu 2
#     # # python train_dual.py StarLightCurves edf_topk15 --loader UCR --batch-size 128 --data_perc 5perc --gpu 2
#     # python train_dual.py ElectricDevices edf_topk15 --loader UCR --batch-size 512 --data_perc 1perc --gpu 2
#     # python trainbk.py ElectricDevices edf_topk15 --loader UCR --batch-size 512 --data_perc 5perc --gpu 2
#     # python train_dual.py StarLightCurves  edf_topk15 --loader UCR --batch-size 64 --data_perc 5perc --gpu 2
#     # python train_dual.py ElectricDevices  edf_topk15 --loader UCR --batch-size 64 --data_perc 5perc --gpu 2
#     # python trainbk.py StarLightCurves  edf_topk15 --loader UCR --batch-size 128 --data_perc 5perc --gpu 2
done


# batch_size=(32 64 128 512)

# for bs in "${batch_size[@]}"
# do
#     python train_dual.py Bridge bs --loader HAR --batch-size $bs --data_perc 5perc --gpu 2
#     python train_dual.py RoadBank bs --loader HAR --batch-size $bs --data_perc 5perc --gpu 2
#     python trainbk.py Bridge bs --loader HAR --batch-size $bs --data_perc 5perc  --gpu 2
#     python trainbk.py RoadBank bs --loader HAR --batch-size $bs --data_perc 5perc  --gpu 2
# done


# temperatures=(0.001 0.01 0.05 0.1 1)

# for temp in "${temperatures[@]}"
# do
#     python train_dual.py Bridge temp --loader HAR --batch-size 256 --data_perc 5perc --gpu 2 --temperature $temp
#     python train_dual.py RoadBank temp --loader HAR --batch-size 256 --data_perc 5perc --gpu 2 --temperature $temp
#     # python trainbk.py Bridge temp --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --temperature $temp
#     # python trainbk.py RoadBank temp --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --temperature $temp
# done 


# ma_gammas=(0.001 0.01 0.05 0.1 1)

# for gammass in "${ma_gammas[@]}"
# do
#     python train_dual.py Bridge gammass --loader HAR --batch-size 256 --data_perc 5perc --gpu 0 --ma_gamma $gammass
#     python train_dual.py RoadBank gammass --loader HAR --batch-size 256 --data_perc 5perc --gpu 0 --ma_gamma $gammass
#     # python trainbk.py Bridge gammass --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --ma_gammas $gammass
#     # python trainbk.py RoadBank gammass --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --ma_gammas $gammass
# done 


# ma_alphas=(0.01 0.1 0.2 0.4 0.6 0.8 1)

# for ma_alpha in "${ma_alphas[@]}"
# do
#     python train_dual.py Bridge ma_alphas --loader HAR --batch-size 256 --data_perc 5perc --gpu 2 --ma_alpha $ma_alpha
#     python train_dual.py RoadBank ma_alphas --loader HAR --batch-size 256 --data_perc 5perc --gpu 2 --ma_alpha $ma_alpha
#     # python trainbk.py Bridge gammass --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --ma_gammas $gammass
#     # python trainbk.py RoadBank gammass --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --ma_gammas $gammass
# done 


# repr_dims=(32 64 128 256 512)

# for dim in "${repr_dims[@]}"
# do
#     python train_dual.py Bridge repr_dims --loader HAR --batch-size 256 --data_perc 5perc --gpu 0 --repr-dims $dim
#     python train_dual.py RoadBank repr_dims --loader HAR --batch-size 256 --data_perc 5perc --gpu 0 --repr-dims $dim
#     # python trainbk.py Bridge gammass --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --ma_gammas $gammass
#     # python trainbk.py RoadBank gammass --loader HAR --batch-size 256 --data_perc 5perc  --gpu 2 --ma_gammas $gammass
# done 