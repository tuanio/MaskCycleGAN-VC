python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SF3_VCC2TF1/ckpts \
    --load_epoch 3000 \
    --model_name generator_A2B

python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TM1 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TM1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SM3_VCC2TM1/ckpts \
    --load_epoch 3000 \
    --model_name generator_A2B

python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TM1 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TM1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SF3_VCC2TM1/ckpts \
    --load_epoch 3000 \
    --model_name generator_A2B

python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TF1 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TF1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SM3_VCC2TF1/ckpts \
    --load_epoch 3000 \
    --model_name generator_A2B 
--- frequency augmented

python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TF1_FreqMask20 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TF1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SF3_VCC2TF1_FreqMask20/ckpts \
    --load_epoch 2200 \
    --model_name generator_A2B

python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TM1_FreqMask20 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TM1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SM3_VCC2TM1_FreqMask20/ckpts \
    --load_epoch 2200 \
    --model_name generator_A2B

python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SF3_VCC2TM1_FreqMask20 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SF3 \
    --speaker_B_id VCC2TM1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SF3_VCC2TM1_FreqMask20/ckpts \
    --load_epoch 2200 \
    --model_name generator_A2B

python3.10 -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_VCC2SM3_VCC2TF1_FreqMask20 \
    --save_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/evaluation \
    --preprocessed_data_dir /data/tuanio/data/vcc2018/preprocessed/evaluation \
    --gpu_ids 0 \
    --speaker_A_id VCC2SM3 \
    --speaker_B_id VCC2TF1 \
    --ckpt_dir /data/tuanio/projects/voice-conversion/mask-cycle-gan-vc/experiments/mask_cyclegan_vc_VCC2SM3_VCC2TF1_FreqMask20/ckpts \
    --load_epoch 2200 \
    --model_name generator_A2B 