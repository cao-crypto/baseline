GPU_ID=2

EMBEDDING_TYPE='gpt35' #word2vec   clip   gpt35 gpt35Diff
NOISE_DIM=1024

DATASET='s3dis'
SPLIT=0
N_WAY=2
K_SHOT=1

WEIGHT1=1
WEIGHT2=1

#DATA_PATH='./datasets/S3DIS/blocks_bs1_s1'
DATA_PATH='../attMPTI-main/datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_'${DATASET}'/GPCPR/'

MODEL_CHECKPOINT=${SAVE_PATH}'S'${SPLIT}'_N'${N_WAY}'_K'${K_SHOT}'_Att1'
PRETRAIN_CHECKPOINT='./log_'${DATASET}'/pretrain_S'${SPLIT}


args=(--phase 'gpcprtrain' --dataset "${DATASET}"
      --n_way $N_WAY --k_shot $K_SHOT --cvfold $SPLIT
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT"
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --use_transformer --use_align --use_supervise_prototype --use_attention
      --use_text --use_text_diff --embedding_type "$EMBEDDING_TYPE" --noise_dim $NOISE_DIM
      --use_pcpr
      --use_dd_loss --dd_ratio1 $WEIGHT1 --dd_ratio2 $WEIGHT2
            )
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"

args=(--phase 'gpcpreval'  --dataset "${DATASET}"
      --n_way $N_WAY --k_shot $K_SHOT --cvfold $SPLIT
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT"
      --data_path  "$DATA_PATH" --save_path "$MODEL_CHECKPOINT"
      --model_checkpoint_path "$MODEL_CHECKPOINT"
      --use_transformer --use_align --use_supervise_prototype --use_attention
      --use_text --use_text_diff --embedding_type "$EMBEDDING_TYPE" --noise_dim $NOISE_DIM
      --use_pcpr
      --use_dd_loss --dd_ratio1 $WEIGHT1 --dd_ratio2 $WEIGHT2
            )
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py "${args[@]}"

