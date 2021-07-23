#!/bin/bash

### CQL walker medium expert

# CLR 3e-5 x ALR {3e-5, 1e-4}, dual with tau 10.0 vs. no dual with alpha 10.0
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_3e-05_dual_True_tau_10.0_alpha_1.0.json &> cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_3e-05_dual_True_tau_10.0_alpha_1.0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_0.0001_dual_True_tau_10.0_alpha_1.0.json &> cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_0.0001_dual_True_tau_10.0_alpha_1.0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_3e-05_dual_False_alpha_10.0.json &> cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_3e-05_dual_False_alpha_10.0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_0.0001_dual_False_alpha_10.0.json &> cql_d4rl_walker2d_medium_expert_clr_0.0003_alr_0.0001_dual_False_alpha_10.0.txt


python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_walker2d_medium_clr_0.0003_alr_0.0001_bsize_100_nepoch_100_alpha_10.0_bc_steps_0.json &> cql_d4rl_ds_walker2d_medium_clr_0.0003_alr_0.0001_bsize_100_nepoch_100_alpha_10.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_halfcheetah_random_clr_0.0003_alr_0.0001_bsize_100_nepoch_100_alpha_10.0_bc_steps_0.json &> cql_d4rl_ds_halfcheetah_random_clr_0.0003_alr_0.0001_bsize_100_nepoch_100_alpha_10.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_walker2d_medium_expert_clr_0.0003_alr_0.0001_bsize_100_nepoch_55_alpha_10.0_bc_steps_0_ld_300_400.json &> cql_d4rl_ds_walker2d_medium_expert_clr_0.0003_alr_0.0001_bsize_100_nepoch_55_alpha_10.0_bc_steps_0_ld_300_400.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_walker2d_medium_expert_clr_0.0003_alr_0.0001_bsize_100_nepoch_55_alpha_5.0_bc_steps_0.json &> cql_d4rl_ds_walker2d_medium_expert_clr_0.0003_alr_0.0001_bsize_100_nepoch_55_alpha_5.0_bc_steps_0.txt


# pen human - BC-Gaussian, CQL (bc, no bc)
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_pen_human_alr_0.0001_bsize_100_nepoch_2000.json &> bc_d4rl_ds_pen_human_alr_0.0001_bsize_100_nepoch_2000.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_pen_human_clr_0.0003_alr_0.0001_bsize_100_nepoch_2000_alpha_10.0_bc_steps_0.json &> cql_d4rl_ds_pen_human_clr_0.0003_alr_0.0001_bsize_100_nepoch_2000_alpha_10.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_pen_human_clr_0.0003_alr_0.0001_bsize_100_nepoch_2000_alpha_10.0_bc_steps_40000.json &> cql_d4rl_ds_pen_human_clr_0.0003_alr_0.0001_bsize_100_nepoch_2000_alpha_10.0_bc_steps_40000.txt


# ant-maze umaze-diverse and medium-diverse, CQL (bc, no bc) x (dual, no dual)
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_0.json &> cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_40000.json &> cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_40000.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_0.json &> cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000.json &> cql_d4rl_ds_antmaze_umaze_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000.txt

python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_0.json &> cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_40000.json &> cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_alpha_10.0_bc_steps_40000.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_0.json &> cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000.json &> cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000.txt

# ant-maze more runs - rnnd 400, 1000
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_umaze_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_d4rl_ds_antmaze_umaze_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_umaze_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True.json &> hbcq_d4rl_ds_antmaze_umaze_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True.txt

python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_False.json &> bc_d4rl_ds_antmaze_umaze_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_umaze_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_d4rl_ds_antmaze_umaze_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_umaze_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True.json &> hbcq_d4rl_ds_antmaze_umaze_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True.txt

python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_medium_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_d4rl_ds_antmaze_medium_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_medium_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True.json &> hbcq_d4rl_ds_antmaze_medium_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True.txt

python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_False.json &> bc_d4rl_ds_antmaze_medium_diverse_plr_0.001_seq_10_mlp__rnnd_1000_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_medium_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_d4rl_ds_antmaze_medium_diverse_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_antmaze_medium_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True.json &> hbcq_d4rl_ds_antmaze_medium_diverse_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True.txt


# pen-human more runs - rnnd 400, 1000
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_2001_gmm_True.json &> bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_2001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_2001_gmm_False.json &> bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_2001_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_pen_human_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_2001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_d4rl_ds_pen_human_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_2001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_pen_human_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_2001.json &> hbcq_d4rl_ds_pen_human_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_2001.txt

python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_2001_gmm_True.json &> bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_2001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_2001_gmm_False.json &> bc_d4rl_ds_pen_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_2001_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_pen_human_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_nepoch_2001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_d4rl_ds_pen_human_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_nepoch_2001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_pen_human_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_nepoch_2001.json &> hbcq_d4rl_ds_pen_human_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_nepoch_2001.txt

# other adroit tasks with bc-rnn
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_door_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_400_nepoch_30_gmm_True.json &> bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_400_nepoch_30_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_400_nepoch_30_gmm_False.json &> bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_400_nepoch_30_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_30_gmm_True.json &> bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_30_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_30_gmm_False.json &> bc_rnn_d4rl_ds_pen_cloned_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_30_gmm_False.txt


# more adroit bc-rnn scan
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_pen_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_door_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_hammer_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_d4rl_ds_relocate_human_plr_0.0001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt


# kitchen exps

# LR 1e-3
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp_300_400_rnnd_400_gmm_False.txt


# LR 1e-4
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_10_mlp_300_400_rnnd_400_gmm_False.txt


# BC normal, LR 1e-3 vs. LR 1e-4, GMM y / n, MLP deep or normal
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_300_400_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_1_mlp_256_256_256_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_300_400_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_True.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.0001_seq_1_mlp_256_256_256_rnn_False_gmm_False.txt


# kitchen - seq scan
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_5_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_5_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_50_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_50_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_100_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_complete_nepoch_1001_horz_281_plr_0.001_seq_100_mlp__rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_5_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_5_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_50_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_50_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_100_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_partial_nepoch_501_horz_281_plr_0.001_seq_100_mlp__rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_5_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_5_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_10_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_50_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_50_mlp__rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl/bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_100_mlp__rnnd_400_gmm_False.json &> bc_rnn_d4rl_ds_kitchen_mixed_nepoch_501_horz_281_plr_0.001_seq_100_mlp__rnnd_400_gmm_False.txt



# ant maze hard with tanh gaussian
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000_tanh_True.json &> cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_100_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000_tanh_True.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_256_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000_tanh_True.json &> cql_d4rl_ds_antmaze_medium_diverse_clr_0.0003_alr_0.0001_bsize_256_nepoch_60_dual_True_tau_5.0_alpha_5.0_bc_steps_40000_tanh_True.txt

# half cheetah random with tanh gaussian
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_halfcheetah_random_clr_0.0003_alr_0.0001_bsize_100_nepoch_100_alpha_5.0_bc_steps_40000_tanh_True.json &> cql_d4rl_ds_halfcheetah_random_clr_0.0003_alr_0.0001_bsize_100_nepoch_100_alpha_5.0_bc_steps_40000_tanh_True.txt
python train.py --algo cql --config ../exps/d4rl/cql_d4rl_ds_halfcheetah_random_clr_0.0003_alr_0.0001_bsize_256_nepoch_100_alpha_5.0_bc_steps_40000_tanh_True.json &> cql_d4rl_ds_halfcheetah_random_clr_0.0003_alr_0.0001_bsize_256_nepoch_100_alpha_5.0_bc_steps_40000_tanh_True.txt


# walker medium expert with hbcq, start critic training late
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_walker2d_medium_expert_clr_0.001_aslr_0.001_aclr_0.0001_c_start_10_actor_True_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp_.json &> hbcq_d4rl_ds_walker2d_medium_expert_clr_0.001_aslr_0.001_aclr_0.0001_c_start_10_actor_True_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp_.txt
python train.py --algo hbcq --config ../exps/d4rl/hbcq_d4rl_ds_walker2d_medium_expert_clr_0.001_aslr_0.001_aclr_0.0001_c_start_10_actor_True_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp_300_400.json &> hbcq_d4rl_ds_walker2d_medium_expert_clr_0.001_aslr_0.001_aclr_0.0001_c_start_10_actor_True_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp_300_400.txt


### Lift RB

# BCQ 4 dataset variants + CQL (no bc) on dense fixed horizon
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_fixed_horz_sparse_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_fixed_horz_sparse_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_task_comp_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_task_comp_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_task_comp_done_timeout_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_task_comp_done_timeout_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo cql --config ../exps/lift_rb/cql_lift_rb_ds_fixed_horz_dense_clr_0.0003_alr_0.0001_bsize_100_alpha_10.0_bc_steps_0.json &> cql_lift_rb_ds_fixed_horz_dense_clr_0.0003_alr_0.0001_bsize_100_alpha_10.0_bc_steps_0.txt

# new 4 dataset variants - BC-GMM (flat) vs. BCQ default
python train.py --algo bc --config ../exps/lift_rb/bc_rnn_lift_rb_ds_dense_os_ts_done_end_plr_0.001_seq_1_rnn_False_gmm_True.json &> bc_rnn_lift_rb_ds_dense_os_ts_done_end_plr_0.001_seq_1_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/lift_rb/bc_rnn_lift_rb_ds_dense_ts_done_end_plr_0.001_seq_1_rnn_False_gmm_True.json &> bc_rnn_lift_rb_ds_dense_ts_done_end_plr_0.001_seq_1_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/lift_rb/bc_rnn_lift_rb_ds_dense_os_ts_done_end_buffer_499_plr_0.001_seq_1_rnn_False_gmm_True.json &> bc_rnn_lift_rb_ds_dense_os_ts_done_end_buffer_499_plr_0.001_seq_1_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/lift_rb/bc_rnn_lift_rb_ds_dense_ts_done_end_buffer_499_plr_0.001_seq_1_rnn_False_gmm_True.json &> bc_rnn_lift_rb_ds_dense_ts_done_end_buffer_499_plr_0.001_seq_1_rnn_False_gmm_True.txt

python train.py --algo bc --config ../exps/lift_rb/bc_rnn_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_plr_0.001_seq_1_rnn_False_gmm_True.json &> bc_rnn_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_plr_0.001_seq_1_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/lift_rb/bc_rnn_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_no_play_plr_0.001_seq_1_rnn_False_gmm_True.json &> bc_rnn_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_no_play_plr_0.001_seq_1_rnn_False_gmm_True.txt


python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_os_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_dense_os_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_dense_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_os_ts_done_end_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_dense_os_ts_done_end_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_end_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_dense_ts_done_end_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt

python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_no_play_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_dense_ts_done_end_buffer_499_zero_act_no_play_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt


# new variants with done mode 2

python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.json &> hbcq_lift_rb_ds_dense_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_os_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.json &> hbcq_lift_rb_ds_dense_os_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_os_ts_done_mode_2_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.json &> hbcq_lift_rb_ds_dense_os_ts_done_mode_2_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.json &> hbcq_lift_rb_ds_dense_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_mode_2_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.json &> hbcq_lift_rb_ds_dense_ts_done_mode_2_buffer_499_clr_0.001_aslr_0.001_aclr_0.0001_actor_True_inf_True.txt

python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_sparse_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_sparse_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_sparse_os_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_sparse_os_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_sparse_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_sparse_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt


# testing late critic epoch start
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True.json &> hbcq_lift_rb_ds_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True.json &> hbcq_lift_rb_ds_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_sparse_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True.json &> hbcq_lift_rb_ds_sparse_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_sparse_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True.json &> hbcq_lift_rb_ds_sparse_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_sparse_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True.json &> hbcq_lift_rb_ds_sparse_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_sparse_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True.json &> hbcq_lift_rb_ds_sparse_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True_inf_True.json &> hbcq_lift_rb_ds_dense_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_100_actor_True_inf_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_dense_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True_inf_True.json &> hbcq_lift_rb_ds_dense_ts_done_mode_2_clr_0.001_aslr_0.001_aclr_0.0001_c_start_100_as_end_-1_actor_True_inf_True.txt


# negative rewards, done mode 1 (fixed horizon), dense vs. sparse, truncate vs. full
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_nr_fixed_horz_sparse_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_nr_fixed_horz_sparse_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_nr_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_nr_fixed_horz_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_nr_dense_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_nr_dense_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/lift_rb/hbcq_lift_rb_ds_nr_sparse_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_lift_rb_ds_nr_sparse_ts_done_end_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt




### RT-v1

# CQL no BC on lift, cans
python train.py --algo cql --config ../exps/rt_v1/cql_rt_v1_ds_lift_subopt_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.json &> cql_rt_v1_ds_lift_subopt_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/rt_v1/cql_rt_v1_ds_cans_top_225_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.json &> cql_rt_v1_ds_cans_top_225_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.txt

python train.py --algo cql --config ../exps/rt_v1/cql_rt_v1_ds_lift_subopt_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_40000.json &> cql_rt_v1_ds_lift_subopt_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_40000.txt
python train.py --algo cql --config ../exps/rt_v1/cql_rt_v1_ds_lift_subopt_clr_0.0003_alr_1e-05_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.json &> cql_rt_v1_ds_lift_subopt_clr_0.0003_alr_1e-05_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.txt

# dense (no BC, BC, LR 1e-5)
python train.py --algo cql --config ../exps/rt_v1/cql_rt_v1_ds_lift_subopt_dense_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.json &> cql_rt_v1_ds_lift_subopt_dense_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.txt
python train.py --algo cql --config ../exps/rt_v1/cql_rt_v1_ds_lift_subopt_dense_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_40000.json &> cql_rt_v1_ds_lift_subopt_dense_clr_0.0003_alr_0.0001_bsize_100_nepoch_801_alpha_10.0_bc_steps_40000.txt
python train.py --algo cql --config ../exps/rt_v1/cql_rt_v1_ds_lift_subopt_dense_clr_0.0003_alr_1e-05_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.json &> cql_rt_v1_ds_lift_subopt_dense_clr_0.0003_alr_1e-05_bsize_100_nepoch_801_alpha_10.0_bc_steps_0.txt


# BC-RNN-VAE with uniform categ latent, beta 0.05 on lift, cans
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_lift_subopt_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_lift_subopt_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# diff KL
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# RNN dim 1000
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# LR 1e-4
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.0001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.0001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# enc-RNN bidirectional
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_True_bidir_True_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_True_bidir_True_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# BC-RNN large hidden dim on cans - LR 1e-3, 1e-4
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__rnnd_400_nepoch_801.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__rnnd_400_nepoch_801.txt
# python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.0001_seq_30_mlp__rnnd_400_nepoch_801.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.0001_seq_30_mlp__rnnd_400_nepoch_801.txt

# BC-RNN large hidden dim (1000) on lift / cans, LR 1e-3
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_lift_subopt_plr_0.001_seq_30_mlp__nepoch_801_rnnd_1000.json &> bc_rnn_rt_v1_ds_lift_subopt_plr_0.001_seq_30_mlp__nepoch_801_rnnd_1000.txt
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__nepoch_801_rnnd_1000.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__nepoch_801_rnnd_1000.txt

# above on cans, but categ + KL 0.5, 0.005
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt


# HBCQ-Latent-Seq

# Lift (dense + sparse, n-step returns)
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_dense_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_dense_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_dense_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_dense_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# scan - rnn hidden dim (400, 1000) + num_epochs (1601, 801) x pc (True, False)

# Lift
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_lift_subopt_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_lift_subopt_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# Cans
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_cans_top_225_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_cans_top_225_latent_True_pc_True_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_400_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_801_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt


# Cans follow-ups with larger hidden dim + longer
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__rnnd_2000_nepoch_1001_gmm_True.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__rnnd_2000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/rt_v1/bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__rnnd_2000_nepoch_1001_gmm_False.json &> bc_rnn_rt_v1_ds_cans_top_225_plr_0.001_seq_30_mlp__rnnd_2000_nepoch_1001_gmm_False.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_1601_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_1601_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_rt_v1_ds_cans_top_225_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_30_rnnd_1000_nepoch_1601_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt

# Cans with HBCQ-RNN-GMM
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_clr_0.001_aslr_0.001_actor_False_seq_30_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1601.json &> hbcq_rt_v1_ds_cans_top_225_clr_0.001_aslr_0.001_actor_False_seq_30_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1601.txt
python train.py --algo hbcq --config ../exps/rt_v1/hbcq_rt_v1_ds_cans_top_225_clr_0.001_aslr_0.001_actor_False_seq_30_as_rnnd_1000_c_rnnd_100_gmm_True_nepoch_1601.json &> hbcq_rt_v1_ds_cans_top_225_clr_0.001_aslr_0.001_actor_False_seq_30_as_rnnd_1000_c_rnnd_100_gmm_True_nepoch_1601.txt




### D4RL Manip - Lift Suboptimal Paired 1 ###

## normal, done at end of trajectory

# BC / BC-RNN / BC-RNN-VAE categ
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_1_rnn_False.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_1_rnn_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.0001_seq_1_rnn_False.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.0001_seq_1_rnn_False.txt

python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_100.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_100.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_400.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_400.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.0001_seq_10_mlp__rnnd_100.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.0001_seq_10_mlp__rnnd_100.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.0001_seq_10_mlp__rnnd_400.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.0001_seq_10_mlp__rnnd_400.txt

python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_rnnd_100_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_rnnd_100_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_rnnd_400_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_rnnd_400_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt


# HBCQ on 4 dataset variants
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_ds_lift_subopt_paired1_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.json &> hbcq_ds_lift_subopt_paired1_dense_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_ds_lift_subopt_paired1_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.json &> hbcq_ds_lift_subopt_paired1_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_dense_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.json &> hbcq_ds_lift_subopt_paired1_dense_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_dense_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.json &> hbcq_ds_lift_subopt_paired1_dense_buffer_300_clr_0.001_aslr_0.001_aclr_0.0001_actor_False.txt

# HBCQ Latent Seq
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.0001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.0001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.0001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.0001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt


# HBCQ-RNN-GMM
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.0001_aslr_0.0001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.0001_aslr_0.0001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.0001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.0001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.0001_aslr_0.0003_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.0001_aslr_0.0003_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.txt


## best runs - on both paired, and careless

# rnnd 400, careless
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_careless_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_careless_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_careless_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_ds_lift_subopt_careless_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_careless_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_careless_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_careless_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.json &> hbcq_ds_lift_subopt_careless_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_nepoch_1001.txt

# rnnd 1000, paired
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.json &> bc_rnn_ds_lift_subopt_paired1_plr_0.001_seq_10_mlp__rnnd_1000_nepoch_1001_gmm_False.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_1000_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_nepoch_1001.txt



# paired - HBCQ-RNN-GMM without mlp layers for rnn
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_400_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_400_gmm_True_mlp_300_400_nepoch_1001.txt


# paired - pre-train action sampler before critic
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_c_start_400_as_end_400_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_c_start_400_as_end_400_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_c_start_400_as_end_-1_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_lift_subopt_paired1_clr_0.001_aslr_0.001_c_start_400_as_end_-1_actor_False_seq_10_as_rnnd_1000_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt

python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_c_start_400_as_end_400_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10.json &> hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.001_aslr_0.001_c_start_400_as_end_400_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.0001_aslr_0.001_c_start_400_as_end_400_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10.json &> hbcq_ds_lift_subopt_paired1_latent_True_pc_False_clr_0.0001_aslr_0.001_c_start_400_as_end_400_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10.txt


# rnnd - (100, 400) - paired 2 with bc-rnn, gmm y/n
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_100_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp__rnnd_400_nepoch_1001_gmm_False.txt

# rnnd 400 - paired 2 with bcq-rnn-gmm, bcq-latent-seq, done 1 vs. 2
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt


# paired 2, same as above but with MLP
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_100_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_100_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_100_nepoch_1001_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_100_nepoch_1001_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_400_nepoch_1001_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_10_mlp_300_400_rnnd_400_nepoch_1001_gmm_False.txt

python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_1_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp_300_400_enc_mlp_300_400_categ_True_ld_1_cd_10.json &> hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp_300_400_enc_mlp_300_400_categ_True_ld_1_cd_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp_300_400_enc_mlp_300_400_categ_True_ld_1_cd_10.json &> hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_mlp_300_400_enc_mlp_300_400_categ_True_ld_1_cd_10.txt


# paired 2, bcq-latent-seq, done 1 vs. 2 - try KL 0.5, 0.005
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired2_done_1_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.json &> hbcq_ds_lift_subopt_paired2_done_2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_mlp__enc_mlp_300_400_categ_True_ld_1_cd_10_plearn_False_enc_rnn_False_dec_rnn_True_enc_sg_True_dec_sg_True.txt


# paired 2, seq length scan
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_5_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_5_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_10_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_10_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_50_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_50_mlp__rnnd_400_nepoch_1001_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_100_mlp__rnnd_400_nepoch_1001_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_100_mlp__rnnd_400_nepoch_1001_gmm_True.txt

python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_5_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_5_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_30_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_30_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_50_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_seq_scan_50_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# paired 2, bc
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_plr_0.0001_seq_scan_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt



# paired 2, stable value / negative rew

# HBCQ normal, gmm / vae, normal / low tau / high num samples
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt

# HBCQ-RNN, normal / low tau / high num samples
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_100_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_100_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_tau_0.0005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_clr_0.001_aslr_0.001_actor_False_tau_0.0005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# same as above, but on dense instead of sparse rewards
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt

python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_100_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_100_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.0005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.0005_nact_samp_10_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# repeat the positive runs with infinite horizon formulation to see if it matches performance of negative
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_10_inf_True.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_10_inf_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_100_inf_True.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005_nact_samp_100_inf_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005_nact_samp_10_inf_True.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005_nact_samp_10_inf_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_10_inf_True.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_10_inf_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_100_inf_True.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005_nact_samp_100_inf_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005_nact_samp_10_inf_True.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005_nact_samp_10_inf_True.txt

python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_10_inf_True_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_10_inf_True_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_100_inf_True_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.005_nact_samp_100_inf_True_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.0005_nact_samp_10_inf_True_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_dense_clr_0.001_aslr_0.001_actor_False_tau_0.0005_nact_samp_10_inf_True_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# paired 2 SUCCESS ONLY, best of {BC, BC-RNN, BCQ, BCQ-RNN} 
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_done_2_succ_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_done_2_succ_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_done_2_succ_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_done_2_succ_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_succ_clr_0.001_aslr_0.001_actor_False_gmm_True_vae_False_tau_0.0005.json &> hbcq_ds_lift_subopt_paired2_done_2_succ_clr_0.001_aslr_0.001_actor_False_gmm_True_vae_False_tau_0.0005.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_done_2_succ_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_lift_subopt_paired2_done_2_succ_clr_0.001_aslr_0.001_actor_False_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# paired 2, image runs

# BC image, LR 1e-3 vs. 1e-4, GMM true vs. false
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_201_rnn_False_gmm_False.txt


# BC image, LR 1e-3 vs. 1e-4, GMM true vs. false, num epochs 501
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.0001_seq_scan_1_mlp_300_400_nepoch_501_rnn_False_gmm_False.txt


# BC-RNN image, LR 1e-3, GMM true vs. false, num epochs 201, batch size 32
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_10_mlp__rnnd_400_nepoch_201_gmm_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_10_mlp__rnnd_400_nepoch_201_gmm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_10_mlp__rnnd_400_nepoch_201_gmm_False.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seq_scan_10_mlp__rnnd_400_nepoch_201_gmm_False.txt



# HBCQ image, defaults, VAE vs. GMM, in memory True
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_image_clr_0.001_aslr_0.001_actor_False_nepoch_501_in_mem_True_gmm_False_vae_True.json &> hbcq_ds_lift_subopt_paired2_image_clr_0.001_aslr_0.001_actor_False_nepoch_501_in_mem_True_gmm_False_vae_True.txt
python train.py --algo hbcq --config ../exps/d4rl_manip/hbcq_ds_lift_subopt_paired2_image_clr_0.001_aslr_0.001_actor_False_nepoch_501_in_mem_True_gmm_True_vae_False.json &> hbcq_ds_lift_subopt_paired2_image_clr_0.001_aslr_0.001_actor_False_nepoch_501_in_mem_True_gmm_True_vae_False.txt

# BC RNN image, LR 1e-3, GMM true, MLP (y /n), RNND (100, 400), in memory True
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_100_nepoch_501_gmm_True_in_mem_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_100_nepoch_501_gmm_True_in_mem_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_400_nepoch_501_gmm_True_in_mem_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_400_nepoch_501_gmm_True_in_mem_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp_300_400_rnnd_100_nepoch_501_gmm_True_in_mem_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp_300_400_rnnd_100_nepoch_501_gmm_True_in_mem_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp_300_400_rnnd_400_nepoch_501_gmm_True_in_mem_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp_300_400_rnnd_400_nepoch_501_gmm_True_in_mem_True.txt

# BC RNN image, LR 1e-3, GMM true - in memory True vs. False
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_400_nepoch_501_gmm_True_in_memm_True.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_400_nepoch_501_gmm_True_in_memm_True.txt
python train.py --algo bc --config ../exps/d4rl_manip/bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_400_nepoch_501_gmm_True_in_memm_False.json &> bc_rnn_ds_lift_subopt_paired2_image_plr_0.001_seqq_10_mlp__rnnd_400_nepoch_501_gmm_True_in_memm_False.txt





# runs on 100 demo v1 datasets

python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/ --done_mode 2 --name states_done_2.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/ --done_mode 2 --shaped --name states_dense_done_2.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/ --done_mode 2 --images --name states_images_done_2.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/ --done_mode 2 --images --shaped --name states_dense_images_done_2.hdf5

python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_pick_place_can/ --done_mode 2 --images --depths --name state_images_depth_done_2.hdf5

python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/ --done_mode 2 --images --depths --name state_images_depth_wrist_cam_done_2.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_1/ --done_mode 2 --images --depths --name state_images_depth_wrist_cam_done_2.hdf5


python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_lift_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_lift_single_expert/ --done_mode 2 --name state_done_2.hdf5

python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_can_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_can_single_expert/ --done_mode 2 --name state_done_2.hdf5

python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_square_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_square_single_expert/ --done_mode 2 --name state_done_2.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_lift_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_120_120.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_can_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_120_120.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_square_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_120_120.hdf5


python merge_hdf5.py --batches \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_wrist_cam_done_2.hdf5 \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_1/state_images_depth_wrist_cam_done_2.hdf5 \
--valid --name state_images_depth_wrist_cam_done_2_all.hdf5

python merge_hdf5.py --batches \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_120_120.hdf5 \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_1/state_images_depth_done_2_120_120.hdf5 \
--valid --name state_images_depth_done_2_120_120_all.hdf5

python merge_hdf5.py --batches \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square/state_images_depth_done_2_84_84.hdf5 \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_1/state_images_depth_done_2_84_84.hdf5 \
--valid --name state_images_depth_done_2_84_84_all.hdf5

python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_done_2.hdf5
python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_dense_done_2.hdf5
python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_images_done_2.hdf5
python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_dense_images_done_2.hdf5

python merge_hdf5.py --batches \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/ajay/states_done_2.hdf5 \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/yuke/states_done_2.hdf5 \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/roberto/states_done_2.hdf5 \
~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_multi/josiah/states_done_2.hdf5 \
--valid --name states_done_2_all.hdf5

# extraction
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square_1/ --done_mode 2 --name states_done_2.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square_1/ --done_mode 2 --shaped --name states_dense_done_2.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square_1/ --done_mode 2 --images --name states_images_done_2.hdf5
python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/sawyer_nut_assembly_square_1/ --done_mode 2 --images --shaped --name states_dense_images_done_2.hdf5

# python demo_to_sars.py --folder ~/Desktop/robosuite_v1_demos/panda_pick_place_can/ --done_mode 2 --images --name states_images_wrist_done_2.hdf5

python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_v1_2/states_done_2.hdf5
python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_v1_2/states_dense_done_2.hdf5
python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_v1_2/states_images_done_2.hdf5
python split_train_val.py --batch ~/Desktop/robosuite_v1_demos/panda_nut_assembly_square_v1_2/states_dense_images_done_2.hdf5

python merge_hdf5.py --batches ~/Desktop/robosuite_v1_demos/panda_pick_place_can_v1_2_1/states_dense_images_done_2.hdf5 ~/Desktop/robosuite_v1_demos/panda_pick_place_can_v1_2_1/states_dense_images_done_2_1.hdf5 --valid --name states_dense_images_done_2_all.hdf5


### Extraction for multi-human datasets for benchmark, using filter keys ("prep_for_release" branch) ###

# postprocess teleop hdf5 and create train-val split for each human's hdf5
python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/chen/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/chen/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/danfei/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/danfei/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/josiah/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/josiah/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/roberto/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/roberto/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/yuke/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/yuke/demo.hdf5


# merge each human's datasets into one hdf5
python merge_hdf5.py --batches \
~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/chen/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/danfei/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/josiah/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/roberto/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/yuke/demo.hdf5 \
--name demo_merged.hdf5 --valid \
--write_filter_keys ajay chen danfei josiah roberto yuke

mv ~/Desktop/benchmark_datasets/panda_lift_multi_human/ajay/demo_merged.hdf5 \
~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5

# create data subsets of interest by grouping different humans together
python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys ajay josiah --name better --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys danfei yuke --name okay --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys chen roberto --name worse --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys danfei yuke ajay josiah --name okay_better --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys chen roberto ajay josiah --name worse_better --valid

python merge_filter_keys.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 \
--filter_keys chen roberto danfei yuke --name worse_okay --valid

# verify dataset looks okay
python get_dataset_info.py \
--dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 --verbose


### demo-to-sars after switching branch to "benchmark" ###
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_lift_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_lift_multi_human/ --done_mode 2 --name state_done_2.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_can_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_can_multi_human/ --done_mode 2 --name state_done_2.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_square_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_square_multi_human/ --done_mode 2 --name state_done_2.hdf5



### Extraction for MART multi-human datasets for benchmark, using filter keys ("prep_for_release" branch) ###

# postprocess teleop hdf5 and create train-val split for each human's hdf5
python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ajay_josiah/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ajay_josiah/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ajay_yuke/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ajay_yuke/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/josiah_chen/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/josiah_chen/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/roberto_chen/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/roberto_chen/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/roberto_danfei/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/roberto_danfei/demo.hdf5

python teleop_to_env_meta.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/yuke_danfei/demo.hdf5
python split_train_val.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/yuke_danfei/demo.hdf5


# merge each human's datasets into one hdf5
python merge_hdf5.py --batches \
~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ajay_josiah/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ajay_yuke/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/josiah_chen/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/roberto_chen/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/roberto_danfei/demo.hdf5 \
~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/yuke_danfei/demo.hdf5 \
--name demo_merged.hdf5 --valid \
--write_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei

mv ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ajay_josiah/demo_merged.hdf5 \
~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5

# # create data subsets of interest by grouping different humans together
# python merge_filter_keys.py \
# --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 \
# --filter_keys ajay josiah --name better --valid

# python merge_filter_keys.py \
# --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 \
# --filter_keys danfei yuke --name okay --valid

# python merge_filter_keys.py \
# --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 \
# --filter_keys chen roberto --name worse --valid

# python merge_filter_keys.py \
# --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 \
# --filter_keys danfei yuke ajay josiah --name okay_better --valid

# python merge_filter_keys.py \
# --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 \
# --filter_keys chen roberto ajay josiah --name worse_better --valid

# python merge_filter_keys.py \
# --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 \
# --filter_keys chen roberto danfei yuke --name worse_okay --valid

sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/hbcq/test_rb_bcq_def_1/hbcq_ds_rb_lift_sparse_done_succ_ld_vae_True_clr_0.001_aslr_0.001_aclr_0.001_actor_False_kl_0.5.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/hbcq/test_rb_bcq_def_1/hbcq_ds_rb_lift_sparse_done_succ_neg_ld_vae_True_clr_0.001_aslr_0.001_aclr_0.001_actor_False_kl_0.5.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/hbcq/test_rb_bcq_def_1/hbcq_ds_rb_lift_dense_done_none_ld_vae_True_clr_0.001_aslr_0.001_aclr_0.001_actor_False_kl_0.5.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/hbcq/test_rb_bcq_1/hbcq_ds_rb_lift_sparse_done_end_ld_actor_True_vae_True_clr_0.001_aslr_0.001_aclr_0.001_kl_0.5_tau_0.005_prior_gmm.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/hbcq/test_rb_bcq_1/hbcq_ds_rb_lift_sparse_done_succ_ld_actor_True_vae_True_clr_0.001_aslr_0.001_aclr_0.001_kl_0.05_tau_0.005_prior_normal.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/hbcq/test_rb_bcq_1/hbcq_ds_rb_lift_sparse_done_succ_ld_actor_True_vae_True_clr_0.001_aslr_0.0001_aclr_0.0001_kl_0.05_tau_0.0005_prior_gmm.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/bc/test_rb_bc_1/bc_ds_rb_lift_sparse_done_end_ld_plr_0.0001_mlp_1024_1024_gmm_False.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/bc/test_rb_bc_rnn_1/bc_rnn_ds_rb_lift_sparse_done_end_ld_plr_0.001_seq_10_rnnd_400_mlp_gmm_False.sbatch
sbatch /cvgl2/u/amandlek/installed_libraries/benchmark/slurm/slurm/scripts/batchrl_benchmark/hp_sweep/low_dim/bc/test_rb_bc_rnn_1/bc_rnn_ds_rb_lift_sparse_done_end_ld_plr_0.0001_seq_10_rnnd_400_mlp_gmm_False.sbatch


python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_dense_done_end.hdf5 --n 12000 --filter_key 12k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_dense_done_none.hdf5 --n 12000 --filter_key 12k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_done_end.hdf5 --n 12000 --filter_key 12k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_done_success.hdf5 --n 12000 --filter_key 12k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_neg_rew_done_end.hdf5 --n 12000 --filter_key 12k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_neg_rew_done_success.hdf5 --n 12000 --filter_key 12k

python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_dense_done_end.hdf5 --n 15000 --filter_key 15k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_dense_done_none.hdf5 --n 15000 --filter_key 15k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_done_end.hdf5 --n 15000 --filter_key 15k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_done_success.hdf5 --n 15000 --filter_key 15k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_neg_rew_done_end.hdf5 --n 15000 --filter_key 15k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/lift_rb_sparse_neg_rew_done_success.hdf5 --n 15000 --filter_key 15k


python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_dense_done_end.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_dense_done_none.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_done_end.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_done_success.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_neg_rew_done_end.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_neg_rew_done_success.hdf5 --n 1500 --filter_key 1.5k

python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_end.hdf5 --n 3900 --filter_key 3.9k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_none.hdf5 --n 3900 --filter_key 3.9k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_end.hdf5 --n 3900 --filter_key 3.9k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_success.hdf5 --n 3900 --filter_key 3.9k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_end.hdf5 --n 3900 --filter_key 3.9k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_success.hdf5 --n 3900 --filter_key 3.9k
 

python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_dense_done_end.hdf5 --n 1800 --filter_key 1.8k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_dense_done_none.hdf5 --n 1800 --filter_key 1.8k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_done_end.hdf5 --n 1800 --filter_key 1.8k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_done_success.hdf5 --n 1800 --filter_key 1.8k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_neg_rew_done_end.hdf5 --n 1800 --filter_key 1.8k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_sparse_neg_rew_done_success.hdf5 --n 1800 --filter_key 1.8k

python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_end.hdf5 --n 5100 --filter_key 5.1k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_none.hdf5 --n 5100 --filter_key 5.1k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_end.hdf5 --n 5100 --filter_key 5.1k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_success.hdf5 --n 5100 --filter_key 5.1k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_end.hdf5 --n 5100 --filter_key 5.1k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_success.hdf5 --n 5100 --filter_key 5.1k
 
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_end.hdf5 --n 5700 --filter_key 5.7k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_none.hdf5 --n 5700 --filter_key 5.7k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_end.hdf5 --n 5700 --filter_key 5.7k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_success.hdf5 --n 5700 --filter_key 5.7k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_end.hdf5 --n 5700 --filter_key 5.7k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_success.hdf5 --n 5700 --filter_key 5.7k
 
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_end.hdf5 --n 6300 --filter_key 6.3k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_dense_done_none.hdf5 --n 6300 --filter_key 6.3k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_end.hdf5 --n 6300 --filter_key 6.3k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_success.hdf5 --n 6300 --filter_key 6.3k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_end.hdf5 --n 6300 --filter_key 6.3k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_neg_rew_done_success.hdf5 --n 6300 --filter_key 6.3k

python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_image_sparse_done_end.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_lift_replay_buffer/rb_image_sparse_done_success.hdf5 --n 1500 --filter_key 1.5k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_image_sparse_done_end.hdf5 --n 5100 --filter_key 5.1k
python truncate_dataset.py --dataset /afs/cs.stanford.edu/u/amandlek/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_image_sparse_done_success.hdf5 --n 5100 --filter_key 5.1k


python train.py --config ~/Downloads/bc_lift_ckpt_fk_1.8k_lr_1e-4_gmm_n_mlp_1024_1024.json 
python train.py --config ~/Downloads/hbcq_lift_ckpt_fk_1.8k_def_actor_y.json  

python train.py --config ~/Downloads/bc_can_ckpt_fk_5.1k_lr_1e-4_gmm_n_mlp_1024_1024.json  
python train.py --config ~/Downloads/hbcq_can_ckpt_fk_5.1k_def_actor_y.json 

python train.py --config ~/Downloads/bc_can_ckpt_fk_5.7k_lr_1e-4_gmm_n_mlp_1024_1024.json  
python train.py --config ~/Downloads/hbcq_can_ckpt_fk_5.7k_def_actor_y.json 

python train.py --config ~/Downloads/bc_can_ckpt_fk_6.3k_lr_1e-4_gmm_n_mlp_1024_1024.json  
python train.py --config ~/Downloads/hbcq_can_ckpt_fk_6.3k_def_actor_y.json 

# verify dataset looks okay
python get_dataset_info.py \
--dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 --verbose


### demo-to-sars after switching branch to "benchmark" ###
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ --done_mode 2 --name state_done_2.hdf5


python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/subopt_can_paired/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/subopt_can_paired/ --done_mode 2 --name state_done_2.hdf5

# obs-extraction for full proprioception to be included in datasets (for observation space ablations)
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_lift_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_lift_single_expert/ --done_mode 2 --name state_done_2_obs.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_can_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_can_single_expert/ --done_mode 2 --name state_done_2_obs.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_square_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_square_single_expert/ --done_mode 2 --name state_done_2_obs.hdf5


python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_lift_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_lift_multi_human/ --done_mode 2 --name state_done_2_obs.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_can_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_can_multi_human/ --done_mode 2 --name state_done_2_obs.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_square_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_square_multi_human/ --done_mode 2 --name state_done_2_obs.hdf5


### TODO: multi-arm on mart_tasks robosuite branch ###
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/ --done_mode 2 --name state_done_2_obs.hdf5

python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ --done_mode 2 --images --depths --name state_images_depth_done_2_84_84_obs.hdf5
python demo_to_sars.py --folder ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/ --done_mode 2 --name state_done_2_obs.hdf5



### prepare dataset ablations - 20% and 50% splits ###
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/demo.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_done_2.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2 

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_single_expert/demo.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_single_expert/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2 

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_single_expert/demo.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_single_expert/state_done_2.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_single_expert/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2 

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/demo.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --name 20_percent --ratio 0.2 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_multi_human/demo.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_multi_human/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_multi_human/demo.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_multi_human/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay josiah yuke roberto danfei chen

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_images_depth_done_2_84_84.hdf5 --name 20_percent --ratio 0.2 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei


python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/demo.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_done_2.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5 

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_single_expert/demo.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_single_expert/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5 

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_single_expert/demo.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_single_expert/state_done_2.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_single_expert/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5 

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/demo.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --name 50_percent --ratio 0.5 
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/demo.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_multi_human/demo.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_can_multi_human/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_multi_human/demo.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_square_multi_human/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay josiah yuke roberto danfei chen

python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/demo.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei
python create_dataset_subsets.py --batch ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_images_depth_done_2_84_84.hdf5 --name 50_percent --ratio 0.5 --input_filter_keys ajay_josiah ajay_yuke josiah_chen roberto_chen roberto_danfei yuke_danfei


# BC normal - 8 runs per dataset, LR {1e-3, 1e-4}, GMM {true / false}, arch {(300, 400), (256, 256, 256)}
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_1001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_1001_rnn_False_gmm_False.txt


# HBCQ normal - 8 runs per dataset - {gmm, vae} x n_samp {10, 100} x tau {0.005, 0.0005}
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_False_vae_True_tau_0.0005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.005_nact_samp_100.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_10.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_nepoch_1001_gmm_True_vae_False_tau_0.0005_nact_samp_100.txt


# BC-RNN - 8 runs per dataset - LR 1e-3, seq 10, gmm {y, n} x rnnd {100, 400} x MLP {y, n}
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_1001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_100_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp_300_400_nepoch_1001_rnnd_400_gmm_False.txt


# BCQ-RNN - 6 runs per dataset. LR 1e-3, seq 10. [rnnd 400, MLP (y / n), tau 0.005], [rnnd (100, 400), MLP n, tau (0.005, 0.0005) ]
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_lift_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_panda_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp_300_400_nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --algo hbcq --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_nut_assembly_square_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_100_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# GTI - 6 runs per dataset - seq 10, PLR / GLR (1e-3, 1e-4), (1e-3, 1e-3) (1e-4, 1e-5), rnnd (100, 400), beta 0.05, MLP n
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt

python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_lift_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_lift_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt

python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt

python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_pick_place_can_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt

python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt

python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_glr_0.0001_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_100_vae_True_kl_0.05.txt
python train.py --algo gti --config ../exps/v1_datasets/gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.json &> gti_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_glr_1e-05_seq_10_nepoch_1001_mlp__rnnd_400_vae_True_kl_0.05.txt


# BC, LR 1e-4, GMM True, more epochs
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.txt


# BC-RNN, LR 1e-4, also more epochs
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_lift_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp_300_400_nepoch_2001_rnnd_400_gmm_False.txt



# nut assembly, BC / BC-RNN with even larger nets
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt


# BC / BC-RNN, GMM variations

# BC
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True_nm_100_min_0.01_act_exp.txt


# BC-RNN
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.01_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_5_min_0.0001_act_exp.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_softplus.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_nm_100_min_0.01_act_exp.txt


# BC image, gmm (y / n), prop (y / n)
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_1_mlp_300_400_nepoch_501_rnn_False_gmm_False_mod_im_prop.txt


# BC-RNN image, gmm (y / n), prop (y / n)
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_lift_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_True_mod_im_prop.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt


# BC-RNN image scans on panda can

# LR 1e-4 (gmm False only)
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0001_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt

# rnn with mlp
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_True_mod_im.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_True_mod_im_prop.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_False_mod_im.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp_300_400_nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt

# rnnd 400
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_True_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_True_mod_im.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_True_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_True_mod_im_prop.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.001_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop.txt

# LR scan + seed variability
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop_seed_0.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop_seed_0.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop_seed_1.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop_seed_1.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop_seed_0.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop_seed_0.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop_seed_1.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0005_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop_seed_1.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0003_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0003_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0003_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0003_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0007_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0007_seq_10_mlp__nepoch_501_rnnd_100_gmm_False_mod_im_prop.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0007_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_image_plr_0.0007_seq_10_mlp__nepoch_501_rnnd_400_gmm_False_mod_im_prop.txt



scp -i ~/.ssh/bc.pem -r ubuntu@3.133.158.11:/home/ubuntu/installed_libraries/batchRL/can_paired2_trained_models/

scp -i ~/.ssh/bc.pem -r ubuntu@3.137.217.215:/home/ubuntu/installed_libraries/batchRL/spirl_trained_models/ .


# nut assembly, 200 demos, BC / BC-RNN

# BC
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_1_mlp_1024_1024_nepoch_2001_rnn_False_gmm_False.txt

# BC-RNN
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt

python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_True.txt
python train.py --algo bc --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_1000_gmm_False.txt


# sawyer can, paired 2

# BC
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_1_mlp_300_400_nepoch_2001_rnn_False_gmm_False.txt

# BC-RNN
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_paired2_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt

# BCQ
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.005.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_False_vae_True_tau_0.0005.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.005.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_nepoch_1601_gmm_True_vae_False_tau_0.0005.txt

# BCQ-RNN
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.0001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.0001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

# BCQ-RNN, more epochs
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.0001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1601.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_clr_0.001_aslr_0.0001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1601.txt

# BCQ-Latent-Seq

# categ, enc-dec subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_categ_True_ld_1_cd_10_pl_False_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt

# N(0, 1), enc-dec subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt

# GMM, enc-dec subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True.txt


# categ, no subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__categ_True_ld_1_cd_10_pl_False_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt

# N(0, 1), no subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt

# GMM, no subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.05_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_clr_0.001_aslr_0.0001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.005_dec_ld__enc_ld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False.txt


# follow-ups - trying single-step return, prior [normal, gmm], KL [0.5, 5e-4], pre-train AS 200 (y / n), enc sg (y / n)

# N(0, 1), subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.5_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.0005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.0005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.0005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_nepoch_1001_vae_True_kl_0.0005_dec_ld__enc_ld_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.txt

# GMM, subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld_300_400_p_gmm_True_p_mlp_300_400_e_rnn_False_d_rnn_True_e_sg_True_d_sg_True_crit_st_200.txt

# N(0, 1), no subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.txt

# GMM, no subgoals
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.5_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_-1.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_latent_True_pc_False_ss_ret_True_clr_0.001_aslr_0.001_actor_False_seq_10_rnnd_400_ne_1001_vae_True_kl_0.0005_dld__eld__p_gmm_True_p_mlp_300_400_e_rnn_True_d_rnn_True_e_sg_False_d_sg_False_crit_st_200.txt


# SPIRL initial scans on panda lift expert
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.5.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.5.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.5.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.5.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.005.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.5.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.5.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.5.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.5.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.005.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.5.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.5.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.5.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.5.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_False_kl_0.005.txt

# more scans

# lower beta
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.0005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.0005.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-05.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-05.txt

# gmm skill prior
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_0.005_min_0.01.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_0.005_min_0.01.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_0.005_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_0.005_min_0.0001.txt

# deeper skill MLP
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_128_128_128_128_128_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_128_128_128_128_128_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_1024_1024_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_1024_1024_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.txt

# encoder bidir
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_bidir_True_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_bidir_True_kl_0.005.txt

# enc + dec MLPs
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld_300_400_dec_ld_300_400_rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.json &> bc_rnn_ds_v1_ds_panda_lift_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld_300_400_dec_ld_300_400_rnnd_400_gaussian_True_gmm_False_lne_True_kl_0.005.txt


# best on panda cans / nut assembly (low beta, gmm true / false)
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-05.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-05.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-06.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-06.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-05_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-05_min_0.0001.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-05.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-05.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-06.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_True_gmm_False_lne_True_kl_5e-06.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-05_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-05_min_0.0001.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt

# best on panda nut assembly 200
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_128_128_128_128_128_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_128_128_128_128_128_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_1024_1024_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.001_slr_0.0001_seq_10_sld_1024_1024_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_300_400_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_128_128_128_128_128_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_128_128_128_128_128_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_1024_1024_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_nepoch_1501_spirl__plr_0.0001_slr_0.0001_seq_10_sld_1024_1024_enc_ld__dec_ld__rnnd_400_gaussian_False_gmm_True_lne_True_kl_5e-06_min_0.0001.txt



# BC-RNN, open-loop

# panda lift
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_lift_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt

# panda can
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt

# sawyer can
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_sawyer_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt

# panda square
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False_open_loop_True.txt


# BC-RNN-VAE, panda can scan

# categ
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True.txt

# N(0, 1)
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True.txt

# GMM - dec cond y
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.txt

# GMM - dec cond n
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False.txt


# BC-RNN-VAE open-loop, panda can scan

# categ
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__categ_True_cd_10_ed_rnn_True_open_loop_True.txt

# N(0, 1)
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__ed_rnn_True_open_loop_True.txt

# GMM - dec cond y
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.txt

# GMM - dec cond n
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.0005_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_dec_cond_False_open_loop_True.txt


# best RNN-VAE runs on panda square 200
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.5_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_vae_True_kl_0.05_mlp__prior_gmm_True_pld_300_400_ed_rnn_True_open_loop_True.txt


# 8-layer RNN on panda square 200
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_128_rnnnl_8_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_128_rnnnl_8_gmm_True.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_128_rnnnl_8_gmm_True.json &> bc_rnn_ds_v1_ds_panda_nut_assembly_square_200_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_128_rnnnl_8_gmm_True.txt


# useful command for moving files from output of grep
ls | grep open_loop | xargs mv -t ../open_loop_trained_models/

ls | grep flow | xargs mv -t ../v1_ds_flow_trained_models/

### OPAL ###
python train.py --config ../exps/d4rl/opal_default_antmaze_medium_diverse.json &> opal_default_antmaze_medium_diverse.txt
python train.py --config ../exps/d4rl/opal_good_antmaze_medium_diverse.json &> opal_good_antmaze_medium_diverse.txt
python train.py --config ../exps/d4rl/opal_default_antmaze_medium_diverse_dec_rnn.json &> opal_default_antmaze_medium_diverse_dec_rnn.txt
python train.py --config ../exps/d4rl/opal_default_antmaze_medium_diverse_no_gauss_dec.json &> opal_default_antmaze_medium_diverse_no_gauss_dec.txt

python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_3e-05_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_1.0_bc_0_gauss_dec_True_kl_0.1.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_3e-05_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_1.0_bc_0_gauss_dec_True_kl_0.1.txt
python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_3e-05_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_1.0_bc_0_gauss_dec_False_kl_0.1.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_3e-05_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_1.0_bc_0_gauss_dec_False_kl_0.1.txt
python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_3e-05_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_1.0_bc_0_gauss_dec_True_kl_0.01.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_3e-05_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_1.0_bc_0_gauss_dec_True_kl_0.01.txt
python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_False_ss_True_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.1.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_False_ss_True_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.1.txt

# more good CQL scans
python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.1_ld_8.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.1_ld_8.txt
python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.1_ld_16.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.1_ld_16.txt
python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.0125_ld_8.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.0125_ld_8.txt
python train.py --config ../exps/d4rl/opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.00625_ld_16.json &> opal_ds_antmaze_medium_diverse_nepoch_201_pretrain_100_clr_0.0003_alr_0.0001_plr_0.001_ns_True_ss_False_gamma_0.99_min_q_5.0_bc_0_gauss_dec_True_kl_0.00625_ld_16.txt


### PLAS ###
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_lat_clip_0.5.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_lat_clip_0.5.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_lat_clip_None.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_lat_clip_None.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_lat_clip_0.5.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_lat_clip_0.5.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_lat_clip_None.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_lat_clip_None.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_lat_clip_0.5.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_lat_clip_0.5.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_lat_clip_None.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_lat_clip_None.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_lat_clip_0.5.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_lat_clip_0.5.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_lat_clip_None.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_lat_clip_None.txt

# better betas hopefully
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_ld_12_kl_0.041666666666666664.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_ld_12_kl_0.041666666666666664.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_ld_16_kl_0.03125.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_ld_16_kl_0.03125.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_ld_12_kl_0.041666666666666664.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_ld_12_kl_0.041666666666666664.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_ld_16_kl_0.03125.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_ld_16_kl_0.03125.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_ld_12_kl_0.041666666666666664.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_ld_12_kl_0.041666666666666664.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_ld_16_kl_0.03125.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_1.0_ld_16_kl_0.03125.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_ld_12_kl_0.041666666666666664.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_ld_12_kl_0.041666666666666664.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_ld_16_kl_0.03125.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_1.0_ld_16_kl_0.03125.txt

# try flow models
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.txt
python train.py --config ../exps/d4rl/plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.json &> plas_ds_walker2d_medium_expert_nepoch_51_pretrain_25_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.txt
python train.py --config ../exps/d4rl/plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.json &> plas_ds_halfcheetah_random_nepoch_101_pretrain_50_lam_0.75_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.txt


### BC-Flow ###

# panda can initial scan
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_2_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_1501_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.txt

# follow-ups
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_3_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_2_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_2_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_3_act_relu.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_256_nf_3_act_tanh.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_1024_1024_8_nf_3_act_tanh.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_tanh.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_8_nf_3_act_tanh.txt

# more follow-ups
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_1024_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_1024_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_1024_nf_4_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_1024_nf_4_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_4_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_4_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_1024_256_nf_3_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_1024_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_1024_256_nf_4_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_1024_256_nf_4_act_relu.txt

python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_8_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_8_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_6_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_6_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_8_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_8_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_6_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_6_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_8_act_relu.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_256_256_oldims_256_256_256_nf_8_act_relu.txt

# more after fixing LL computation
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_relu_grad_1e5.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_relu_grad_1e5.txt
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_tanh_grad_1e5.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_tanh_grad_1e5.txt
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_relu.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_tanh.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_1024_1024_256_nf_6_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_4_act_relu.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_4_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_4_act_tanh.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_1024_oldims_1024_1024_256_nf_4_act_tanh.txt
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_3_act_relu.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_3_act_relu.txt
python train.py --config ../exps/v1_datasets/bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_3_act_tanh.json &> bc_rnn1_ds_v1_ds_panda_pick_place_can_plr_0.0001_seq_1_nepoch_2201_flow_True_ldims_1024_1024_oldims_256_256_256_nf_3_act_tanh.txt


### Cans Paired 2 Follow-ups ###

# hbcq latent seq normal
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.005_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.005_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_f_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_f_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t.txt

# hbcq latent seq open-loop
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_300_400_ernn_t_drnn_t_oloop_True.txt

# hbcq spirl
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.txt

python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.txt

python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.txt

python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_f.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_f_tr_f.txt

# best hbcq spirl, more variants
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_f_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_f_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-07_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-07_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_crit_st_200_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_crit_st_200_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/v1_datasets/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_128_128_128_128_128_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_128_128_128_128_128_gmm_True_min_0.0001_lne_t_tr_t.txt


# hbcq latent seq with flow prior
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_256_256_nf_2_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_256_256_nf_2_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_2_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_2_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_256_256_nf_3_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_256_256_nf_3_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_3_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_3_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_256_256_nf_2_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_256_256_nf_2_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_2_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_2_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_256_256_nf_3_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_256_256_nf_3_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_3_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_f_fld_1024_1024_nf_3_ernn_t_drnn_t.txt

python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_256_256_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_256_256_256_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_1024_1024_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_1024_1024_256_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_256_256_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_256_256_256_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_1024_1024_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_1024_1024_256_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_256_256_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_256_256_256_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_1024_1024_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_2_p_mlp_1024_1024_256_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_256_256_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_256_256_256_ernn_t_drnn_t.txt
python train.py --config ../exps/cans_paired2/hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_1024_1024_256_ernn_t_drnn_t.json &> hbcq_ds_v1_ds_sawyer_pick_place_can_paired2_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_flow_t_c_t_fld_256_256_nf_3_p_mlp_1024_1024_256_ernn_t_drnn_t.txt


### HAN-VAE ###

# initial beta scan
python train.py --config ../exps/han/bc_han_vae_panda_lift_lr_1e-3_beta_5e-2.json &> bc_han_vae_panda_lift_lr_1e-3_beta_5e-2.txt
python train.py --config ../exps/han/bc_han_vae_panda_lift_lr_1e-3_beta_5e-3.json &> bc_han_vae_panda_lift_lr_1e-3_beta_5e-3.txt
python train.py --config ../exps/han/bc_han_vae_panda_lift_lr_1e-3_beta_5e-5.json &> bc_han_vae_panda_lift_lr_1e-3_beta_5e-5.txt
python train.py --config ../exps/han/bc_han_vae_panda_lift_lr_1e-3_beta_5e-6.json &> bc_han_vae_panda_lift_lr_1e-3_beta_5e-6.txt

# more runs
python train.py --config ../exps/han/bc_han_vae_panda_lift_lr_1e-3_beta_5e-3_temp_10_categ_49_ne_1001.json &> bc_han_vae_panda_lift_lr_1e-3_beta_5e-3_temp_10_categ_49_ne_1001.txt 
python train.py --config ../exps/han/bc_han_vae_panda_can_lr_1e-3_beta_5e-3_temp_10_categ_49_ne_1001.json &> bc_han_vae_panda_can_lr_1e-3_beta_5e-3_temp_10_categ_49_ne_1001.txt 
python train.py --config ../exps/han/bc_han_vae_panda_can_lr_1e-3_beta_5e-2_temp_10_categ_49_ne_501.json &> bc_han_vae_panda_can_lr_1e-3_beta_5e-2_temp_10_categ_49_ne_501.txt 
python train.py --config ../exps/han/bc_han_vae_panda_can_lr_1e-3_beta_5e-4_temp_10_categ_49_ne_501.json &> bc_han_vae_panda_can_lr_1e-3_beta_5e-4_temp_10_categ_49_ne_501.txt 

# landmark initial runs
python train.py --config ../exps/han/bc_han_landmark_panda_lift_lr_1e-3.json &> bc_han_landmark_panda_lift_lr_1e-3.txt
python train.py --config ../exps/han/bc_han_landmark_panda_lift_lr_1e-3_fixed_var_0.05.json &> bc_han_landmark_panda_lift_lr_1e-3_fixed_var_0.05.txt
python train.py --config ../exps/han/bc_han_landmark_panda_lift_lr_1e-3_fixed_var_0.05_thresh_0.1.json &> bc_han_landmark_panda_lift_lr_1e-3_fixed_var_0.05_thresh_0.1.txt
python train.py --config ../exps/han/bc_han_landmark_panda_can_lr_1e-3_fixed_var_0.05.json &> bc_han_landmark_panda_can_lr_1e-3_fixed_var_0.05.txt

python train.py --config ../exps/han/bc_han_landmark_panda_can_lr_1e-3_fixed_var_0.05_thresh_0.1.json &> bc_han_landmark_panda_can_lr_1e-3_fixed_var_0.05_thresh_0.1.txt
python train.py --config ../exps/han/bc_rnn_ds_v1_ds_panda_pick_place_can_proprio_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_proprio_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_True.txt
python train.py --config ../exps/han/bc_rnn_ds_v1_ds_panda_pick_place_can_proprio_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_False.json &> bc_rnn_ds_v1_ds_panda_pick_place_can_proprio_plr_0.0001_seq_1_mlp_256_256_256_nepoch_2001_rnn_False_gmm_False.txt

# v1.2 robosuite panda can sanity check
python train.py --config ../exps/v1_2_datasets/bc_rnn_ds_v1_2_panda_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_2_panda_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --config ../exps/v1_2_datasets/bc_rnn_ds_v1_2_panda_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_2_panda_can_plr_0.001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt
python train.py --config ../exps/v1_2_datasets/bc_rnn_ds_v1_2_panda_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.json &> bc_rnn_ds_v1_2_panda_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_True.txt
python train.py --config ../exps/v1_2_datasets/bc_rnn_ds_v1_2_panda_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.json &> bc_rnn_ds_v1_2_panda_can_plr_0.0001_seq_10_mlp__nepoch_2001_rnnd_400_gmm_False.txt


# BC with crop augmentation
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_crop_t_s_60_n_20_coord_t.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_crop_t_s_60_n_20_coord_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_crop_t_s_60_n_20_coord_f.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_crop_t_s_60_n_20_coord_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_t.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_f.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_t.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_t_pt_t.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_t_pt_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_t.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_t.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_coord_t.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_coord_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_pt_t.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_501_mod_im_prop_pt_t.txt

# more
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_t_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_t_pt_f.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_t_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_t_pt_f.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_t.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_t.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_t.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_t.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_f.txt

# wrist
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_f_cat_t.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_f_cat_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_crop_t_s_60_n_20_coord_f_pt_f_cat_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_f_cat_t.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_f_cat_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_coord_f_pt_f_cat_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_cat_t.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_cat_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_cat_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_f_cat_t.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_f_cat_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_f_cat_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_t_cat_t.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_t_cat_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_t_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_coord_f_pt_t_cat_f.txt

# next batch of runs

# crop pos enc
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_t.json &> bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_t.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_t.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_True_coord_f_pt_f_cat_f.txt

# lr 1e-3 instead of 1e-4 on best runs
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f.txt

# BC wrist (no crop) with coord conv
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_coord_t_pt_t_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_coord_t_pt_t_cat_f.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_t_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_t_cat_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_coord_t_pt_t_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_coord_t_pt_t_cat_f.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_t_pt_t_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_t_pt_t_cat_f.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_cat_f.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_t_pt_f_cat_f.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_t_pt_f_cat_f.txt

# same as tars / capri
python train.py --config ../exps/han/bc_han_landmark_panda_can_lr_1e-4_temp_10.0_fixed_var_0.05_thresh_0.3_n_2_ne_1002.json &> bc_han_landmark_panda_can_lr_1e-4_temp_10.0_fixed_var_0.05_thresh_0.3_n_2_ne_1002.txt
python train.py --config ../exps/han/bc_han_landmark_panda_can_lr_1e-4_temp_10.0_fixed_var_0.05_thresh_0.3_n_2_ne_1002_no_prop.json &> bc_han_landmark_panda_can_lr_1e-4_temp_10.0_fixed_var_0.05_thresh_0.3_n_2_ne_1002_no_prop.txt


# some ablations

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_ssmax_t_no_lin_temp_1.0.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_ssmax_t_no_lin_temp_1.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_ssmax_t_temp_10.0.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_ssmax_t_temp_10.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_ssmax_f_temp_1.0.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_ssmax_f_temp_1.0.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_True_cat_f_ssmax_t_no_lin_temp_1.0.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_True_cat_f_ssmax_t_no_lin_temp_1.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_True_cat_f_ssmax_t_temp_10.0.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_True_cat_f_ssmax_t_temp_10.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_True_cat_f_ssmax_f_temp_1.0.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_True_cat_f_ssmax_f_temp_1.0.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_t_pt_f_ssmax_t_no_lin_temp_1.0.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_t_pt_f_ssmax_t_no_lin_temp_1.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_t_pt_f_ssmax_t_temp_10.0.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_t_pt_f_ssmax_t_temp_10.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_t_pt_f_ssmax_f_temp_1.0.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_t_pt_f_ssmax_f_temp_1.0.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_ssmax_t_no_lin_temp_1.0.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_ssmax_t_no_lin_temp_1.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_ssmax_t_temp_10.0.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_ssmax_t_temp_10.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_ssmax_f_temp_1.0.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_ssmax_f_temp_1.0.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_ssmax_t_no_lin_temp_1.0.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_ssmax_t_no_lin_temp_1.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_ssmax_t_temp_10.0.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_ssmax_t_temp_10.0.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_ssmax_f_temp_1.0.json &> bc_ds_v1_panda_can_image_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_ssmax_f_temp_1.0.txt

python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_gmm_True.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_crop_t_s_60_n_20_pos_enc_False_coord_f_pt_f_cat_f_gmm_True.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_cat_f_gmm_True.json &> bc_ds_v1_panda_can_image_wrist_plr_0.0001_nepoch_1001_mod_im_prop_coord_f_pt_t_cat_f_gmm_True.txt
python train.py --config ../exps/han/bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_gmm_True.json &> bc_ds_v1_panda_can_image_plr_0.001_nepoch_1001_mod_im_prop_coord_t_pt_f_gmm_True.txt




### low-dim runs on multi-human ###

# BC, can
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt


# BC-RNN, can
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt


# HBCQ-Seq, can
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_can_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_can_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_can_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_can_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_can_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_can_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_can_all_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_can_all_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0001_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0001_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_can_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_can_all_clr_0.001_aslr_0.0001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_can_all_clr_0.001_aslr_0.0001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# BC, square
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.json &> bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.json &> bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_1_mlp_300_400_nepoch_1501_rnn_False_gmm_False.txt

# BC-RNN, square
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.json &> bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.json &> bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_400_gmm_False.txt


# HBCQ-Seq, square
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_square_roberto_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_square_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_square_roberto_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_square_yuke_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_square_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_square_yuke_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-06_dld__eld__ernn_t_drnn_t_spirl__slr_0.0001_sld_1024_1024_gmm_True_min_0.0001_lne_t_tr_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_square_all_clr_0.001_aslr_0.001_actor_False_tau_0.005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_square_all_clr_0.001_aslr_0.001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt

python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0001_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0001_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_5e-05_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.json &> hbcq_ds_multi_panda_square_all_lt_t_ss_t_clr_0.001_aslr_0.0001_actor_f_seq_10_rnnd_400_ne_1001_kl_0.0005_dld__eld__p_gmm_t_p_mlp_1024_1024_ernn_t_drnn_t.txt
python train.py --config ../exps/multi/hbcq_ds_multi_panda_square_all_clr_0.001_aslr_0.0001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.json &> hbcq_ds_multi_panda_square_all_clr_0.001_aslr_0.0001_actor_False_tau_0.0005_seq_10_as_rnnd_400_c_rnnd_100_gmm_True_mlp__nepoch_1001.txt


# BC-RNN, rnnd 1000, cans
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_can_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt

# BC-RNN, rnnd 1000, square

python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_ajay_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_josiah_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_roberto_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_yuke_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_all_plr_0.001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.json &> bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_True.txt
python train.py --config ../exps/multi/bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.json &> bc_rnn_ds_multi_panda_square_all_plr_0.0001_seq_10_mlp__nepoch_1501_rnnd_1000_gmm_False.txt


# scp -i ~/.ssh/bc.pem -r ubuntu@3.15.149.29:/home/ubuntu/installed_libraries/batchRL/multi_trained_models/


# han-new runs
python train.py --config ../exps/han_new/.json &> .txt

python train.py --config ../exps/han_new/han_stack_eval_t_depth_fix.json &> han_stack_eval_t_depth_fix.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_f_only_crops.json &> han_chen_stack_eval_f_only_crops.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_f_only_crops_L2.json &> han_chen_stack_eval_f_only_crops_L2.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_only_crops_L2.json &> han_chen_stack_eval_t_only_crops_L2.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix.json &> han_chen_stack_eval_t_depth_fix.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ee_crop_0.0.json &> han_chen_stack_eval_t_depth_fix_ee_crop_0.0.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ee_crop_1.0.json &> han_chen_stack_eval_t_depth_fix_ee_crop_1.0.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_decay_n_cos_n.json &> han_chen_stack_eval_t_depth_fix_decay_n_cos_n.txt

python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_decay_n_cos_n_loss_reweight_n.json &> han_chen_stack_eval_t_depth_fix_decay_n_cos_n_loss_reweight_n.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_decay_n_cos_y_loss_reweight_n.json &> han_chen_stack_eval_t_depth_fix_decay_n_cos_y_loss_reweight_n.txt

python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ee_crop_1.0_no_test_rand.json &> han_chen_stack_eval_t_depth_fix_ee_crop_1.0_no_test_rand.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ee_crop_0.4_no_test_rand.json &> han_chen_stack_eval_t_depth_fix_ee_crop_0.4_no_test_rand.txt

python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_no_control_law.json &> han_chen_stack_eval_t_depth_fix_no_control_law.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_f_depth_fix_no_control_law.json &> han_chen_stack_eval_f_depth_fix_no_control_law.txt

python train.py --config ../exps/han_new/han_chen_stack_eval_f_depth_fix_no_control_law_no_eef.json &> han_chen_stack_eval_f_depth_fix_no_control_law_no_eef.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_no_control_law_no_eef.json &> han_chen_stack_eval_t_depth_fix_no_control_law_no_eef.txt


# han-new ablation runs
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_pos_f.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_pos_f.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_crop_f.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_crop_f.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_crop_f_ee_pos_f.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_crop_f_ee_pos_f.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_conf_f_ee_pos_f.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_conf_f_ee_pos_f.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_crop_f_conf_f_ee_pos_f.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_crop_f_conf_f_ee_pos_f.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ee_pos_f.json &> han_chen_stack_eval_t_depth_fix_ee_pos_f.txt

python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_hard_att_temp_1.0.json &> han_chen_stack_eval_t_depth_fix_hard_att_temp_1.0.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_hard_att_temp_10.0.json &> han_chen_stack_eval_t_depth_fix_hard_att_temp_10.0.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_pos_f_hard_att_temp_1.0.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_pos_f_hard_att_temp_1.0.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_pos_f_hard_att_temp_10.0.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_ee_pos_f_hard_att_temp_10.0.txt

python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_crop_t_s_119_n_1.json &> han_chen_stack_eval_t_depth_fix_crop_t_s_119_n_1.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_crop_t_s_108_n_1.json &> han_chen_stack_eval_t_depth_fix_crop_t_s_108_n_1.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_crop_f.json &> han_chen_stack_eval_t_depth_fix_crop_f.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_crop_t_s_108_n_1.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_crop_t_s_108_n_1.txt
python train.py --config ../exps/han_new/han_chen_stack_eval_t_depth_fix_ctrl_law_f_crop_f.json &> han_chen_stack_eval_t_depth_fix_ctrl_law_f_crop_f.txt


# bc crops x image size
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt

python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt

python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_t_h_30_w_30_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_can_im_60_60_crop_t_h_30_w_30_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_can_im_60_60_crop_t_h_30_w_30_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.json &> bc_ds_v1_panda_can_im_60_60_crop_t_h_30_w_30_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_False.txt

### TODO: 3 of the runs require single machines ###
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt

python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt

python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_60_60_crop_f_h_60_w_60_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_60_60_crop_t_h_30_w_30_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_60_60_crop_t_h_30_w_30_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_gmm_True.txt

# follow-up rnn
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_60_60_crop_t_h_56_w_56_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt

python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_can_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt


# same as above, but panda square 200
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt

python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_120_120_crop_f_h_120_w_120_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_120_120_crop_t_h_108_w_108_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_120_120_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt

python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_84_84_crop_f_h_84_w_84_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_84_84_crop_f_h_84_w_84_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_84_84_crop_t_h_76_w_76_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_84_84_crop_t_h_76_w_76_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_ds_v1_panda_square_all_im_84_84_crop_t_h_42_w_42_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.json &> bc_ds_v1_panda_square_all_im_84_84_crop_t_h_42_w_42_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_gmm_True.txt

# rnn
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_square_all_im_84_84_crop_f_h_84_w_84_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_square_all_im_84_84_crop_f_h_84_w_84_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_square_all_im_84_84_crop_t_h_76_w_76_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_square_all_im_84_84_crop_t_h_76_w_76_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_square_all_im_84_84_crop_t_h_42_w_42_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_square_all_im_84_84_crop_t_h_42_w_42_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt

python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt
python train.py --config ../exps/han_image_size/bc_rnn_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.json &> bc_rnn_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_gmm_True.txt

# HAN panda cans initial runs
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160.json &> han_ds_v1_panda_can_im_120_160.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_ctrl_law_f.json &> han_ds_v1_panda_can_im_120_160_ctrl_law_f.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_ctrl_law_f_ee_pos_f.json &> han_ds_v1_panda_can_im_120_160_ctrl_law_f_ee_pos_f.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_ctrl_law_f_hard_att_temp_1.0.json &> han_ds_v1_panda_can_im_120_160_ctrl_law_f_hard_att_temp_1.0.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_ctrl_law_f_ee_pos_f_hard_att_temp_1.0.json &> han_ds_v1_panda_can_im_120_160_ctrl_law_f_ee_pos_f_hard_att_temp_1.0.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_ctrl_law_f_crop_t_s_108_n_1.json &> han_ds_v1_panda_can_im_120_160_ctrl_law_f_crop_t_s_108_n_1.txt

# sanity check on wrist y/n gmm y/n crops 1/10 using han config - results were weird before
python train.py --config ../exps/han_image_size/ &> 

python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4_wrist.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4_wrist.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4_gmm.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4_gmm.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4_wrist_gmm.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_10_LR_1e-4_wrist_gmm.txt

python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4_gmm.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4_gmm.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4_wrist.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4_wrist.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4_wrist_gmm.json &> han_ds_v1_panda_can_im_120_160_no_action_space_crops_1_LR_1e-4_wrist_gmm.txt

# more HAN action space scans
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_eef_crop_test.json &> han_ds_v1_panda_can_im_120_160_eef_crop_test.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_LR_1e-3.json &> han_ds_v1_panda_can_im_120_160_LR_1e-3.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_eef_pos.json &> han_ds_v1_panda_can_im_120_160_no_eef_pos.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_no_eef_crop.json &> han_ds_v1_panda_can_im_120_160_no_eef_crop.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_offset_0.25_all.json &> han_ds_v1_panda_can_im_120_160_offset_0.25_all.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_gain_bias_0.5_scale_0.5.json &> han_ds_v1_panda_can_im_120_160_gain_bias_0.5_scale_0.5.txt


# HAN 3d policy - initial runs
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_n.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_n.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_1_rel_y_eef_n_crop_10_eef_crop_n.json &> han_ds_v1_panda_can_im_120_160_3d_n_1_rel_y_eef_n_crop_10_eef_crop_n.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_10_rel_y_eef_n_crop_10_eef_crop_n.json &> han_ds_v1_panda_can_im_120_160_3d_n_10_rel_y_eef_n_crop_10_eef_crop_n.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_1_eef_crop_n.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_1_eef_crop_n.txt

python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_y_crop_10_eef_crop_n.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_y_crop_10_eef_crop_n.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_n_eef_n_crop_10_eef_crop_n.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_n_eef_n_crop_10_eef_crop_n.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n.txt

# ablations on best initial run
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201_LR_3e-4.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201_LR_3e-4.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201_LR_1e-3.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201_LR_1e-3.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_n_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_n_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_y_crop_10_eef_crop_y_grip_n_epoch_1201.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_y_crop_10_eef_crop_y_grip_n_epoch_1201.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_1_eef_crop_y_grip_n_epoch_1201.json &> han_ds_v1_panda_can_im_120_160_3d_n_3_rel_y_eef_n_crop_1_eef_crop_y_grip_n_epoch_1201.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_10_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.json &> han_ds_v1_panda_can_im_120_160_3d_n_10_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_n_1_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.json &> han_ds_v1_panda_can_im_120_160_3d_n_1_rel_y_eef_n_crop_10_eef_crop_y_grip_n_epoch_1201.txt 

# ablation with wrist obs
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_n.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_y.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_n_crop_10_eef_crop_y_grip_y.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_n_crop_1_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_n_crop_1_eef_crop_y_grip_n.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_n_crop_10_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_n_crop_10_eef_crop_y_grip_n.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_1_rel_y_eef_n_crop_10_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_1_rel_y_eef_n_crop_10_eef_crop_y_grip_n.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_y_crop_10_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_3_rel_y_eef_y_crop_10_eef_crop_y_grip_n.txt 

# more n keypoints on best runs
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_y_crop_10_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_y_crop_10_eef_crop_y_grip_n.txt 
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_30_rel_y_eef_y_crop_10_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_30_rel_y_eef_y_crop_10_eef_crop_y_grip_n.txt

python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_n_crop_1_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_n_crop_1_eef_crop_y_grip_n.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_30_rel_y_eef_n_crop_1_eef_crop_y_grip_n.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_30_rel_y_eef_n_crop_1_eef_crop_y_grip_n.txt 

python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_n_crop_10_eef_crop_y_grip_y.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_10_rel_y_eef_n_crop_10_eef_crop_y_grip_y.txt
python train.py --config ../exps/han_image_size/han_ds_v1_panda_can_im_120_160_3d_wrist_n_30_rel_y_eef_n_crop_10_eef_crop_y_grip_y.json &> han_ds_v1_panda_can_im_120_160_3d_wrist_n_30_rel_y_eef_n_crop_10_eef_crop_y_grip_y.txt 


# BC-spirl with image, KL scan + randomization scan
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-05.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-05.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.txt

python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-05.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-05.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.txt

python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-05.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-05.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.txt

# rnnd 1000
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_0.0005.txt

# vae-rnn with proprio
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05_prop_t.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05_prop_t.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_prop_t.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_prop_t.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_prop_t.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_prop_t.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05_prop_t.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.05_prop_t.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_prop_t.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_prop_t.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_prop_t.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_prop_t.txt

# on panda square 200
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06.txt

# 3d conditioning

# (no crop - not run)
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_True.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_True.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_False.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_False.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_True.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_True.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.txt

# crop 1
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_True.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_True.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_False.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_False.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_True.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_True.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.txt


# vae-rnn with proprio, panda square
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_prop_t.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_prop_t.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_prop_t.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_prop_t.txt

# rnnd 1000, panda square
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_f_h_120_w_160_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_5e-06.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_1000_kl_5e-06.txt

# 3d conditioning, panda square
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_False.json &> bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_3_eef_False.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_3_eef_False.json &> bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_3_eef_False.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_10_eef_False.txt


# hierarchical SPIRL
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.txt

python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_301_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.txt


# re-do some runs
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_10_eef_False.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_square_all_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_5e-06_nkp_10_eef_False.txt

python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.txt
python train.py --config ../exps/han_image_size/bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.json &> bc_spirl_3d_ds_v1_panda_can_im_120_160_crop_t_h_60_w_60_n_10_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_10_mlp__rnnd_400_kl_0.0005_nkp_10_eef_False.txt

python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_square_all_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.txt

python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_10_h_5_mlp__rnnd_400_kl_5e-06.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_0.0005.txt
python train.py --config ../exps/han_image_size/bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.json &> bc_spirl_ds_v1_panda_can_im_120_160_crop_t_h_108_w_144_n_1_plr_0.0001_nepoch_601_mod_im_wrist_prop_cat_f_seq_50_l_5_h_10_mlp__rnnd_400_kl_5e-06.txt


# HITL new runs

# image
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_base_plr_0.001.json &> bc_rnn_im_ds_base_plr_0.001.txt
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_base_plr_0.0001.json &> bc_rnn_im_ds_base_plr_0.0001.txt
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_more_plr_0.001.json &> bc_rnn_im_ds_more_plr_0.001.txt
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_more_plr_0.0001.json &> bc_rnn_im_ds_more_plr_0.0001.txt

python train.py --config ../exps/hitl_image/bc_rnn_im_ds_v3_hg_dagger_plr_0.0001_iwr_True.json &> bc_rnn_im_ds_v3_hg_dagger_plr_0.0001_iwr_True.txt
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_v3_hg_dagger_plr_0.0001_iwr_False.json &> bc_rnn_im_ds_v3_hg_dagger_plr_0.0001_iwr_False.txt
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_v3_iwr_plr_0.0001_iwr_True.json &> bc_rnn_im_ds_v3_iwr_plr_0.0001_iwr_True.txt
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_v3_iwr_plr_0.0001_iwr_False.json &> bc_rnn_im_ds_v3_iwr_plr_0.0001_iwr_False.txt

python train.py --config ../exps/hitl_image/bc_rnn_im_ds_v3_hg_dagger_cross_plr_0.0001_iwr_True.json &> bc_rnn_im_ds_v3_hg_dagger_cross_plr_0.0001_iwr_True.txt
python train.py --config ../exps/hitl_image/bc_rnn_im_ds_v3_hg_dagger_cross_plr_0.0001_iwr_False.json &> bc_rnn_im_ds_v3_hg_dagger_cross_plr_0.0001_iwr_False.txt


# low-dim
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_base_plr_0.001.json &> bc_rnn_ld_ds_base_plr_0.001.txt
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_base_plr_0.0001.json &> bc_rnn_ld_ds_base_plr_0.0001.txt
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_more_plr_0.001.json &> bc_rnn_ld_ds_more_plr_0.001.txt
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_more_plr_0.0001.json &> bc_rnn_ld_ds_more_plr_0.0001.txt

python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_v3_hg_dagger_plr_0.0001_iwr_True.json &> bc_rnn_ld_ds_v3_hg_dagger_plr_0.0001_iwr_True.txt
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_v3_hg_dagger_plr_0.0001_iwr_False.json &> bc_rnn_ld_ds_v3_hg_dagger_plr_0.0001_iwr_False.txt
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_v3_iwr_plr_0.0001_iwr_True.json &> bc_rnn_ld_ds_v3_iwr_plr_0.0001_iwr_True.txt
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_v3_iwr_plr_0.0001_iwr_False.json &> bc_rnn_ld_ds_v3_iwr_plr_0.0001_iwr_False.txt

python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_v3_hg_dagger_cross_plr_0.0001_iwr_True.json &> bc_rnn_ld_ds_v3_hg_dagger_cross_plr_0.0001_iwr_True.txt
python train.py --config ../exps/hitl_low_dim/bc_rnn_ld_ds_v3_hg_dagger_cross_plr_0.0001_iwr_False.json &> bc_rnn_ld_ds_v3_hg_dagger_cross_plr_0.0001_iwr_False.txt




# dataset playback for RT benchmark

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_replay_buffer/rb_sparse_done_success.hdf5 --filter_key 3.9k --video_path ~/Downloads/playback_can_rb_3.9k.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/subopt_can_paired/state_done_2.hdf5 --video_path ~/Downloads/playback_can_paired.mp4 --n 50

# worse (roberto, chen)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key roberto --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_worse_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key chen --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_worse_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key roberto --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_worse_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key chen --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_worse_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key roberto --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_worse_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key chen --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_worse_2.mp4

# okay (yuke, danfei)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key yuke --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_okay_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key danfei --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_okay_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key yuke --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_okay_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key danfei --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_okay_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key yuke --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_okay_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key danfei --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_okay_2.mp4

# better (ajay, josiah)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key ajay --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_better_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_multi_human/state_done_2.hdf5 --filter_key josiah --video_path ~/Downloads/rt_benchmark_playback/playback_lift_mh_better_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key ajay --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_better_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_multi_human/state_done_2.hdf5 --filter_key josiah --video_path ~/Downloads/rt_benchmark_playback/playback_can_mh_better_2.mp4

python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key ajay --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_better_1.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_multi_human/state_done_2.hdf5 --filter_key josiah --video_path ~/Downloads/rt_benchmark_playback/playback_square_mh_better_2.mp4


# 6 subsets for transport
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key ajay_josiah --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_better.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key yuke_danfei --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_okay.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key roberto_chen --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_worse.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key ajay_yuke --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_okay_better.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key roberto_danfei --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_worse_okay.mp4
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_multi_human/state_done_2.hdf5 --filter_key josiah_chen --video_path ~/Downloads/rt_benchmark_playback/playback_transport_mh_worse_better.mp4

# SE
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_lift_se.mp4 --n 50
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_can_se.mp4 --n 50
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_square_se.mp4 --n 50
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_transport_se.mp4 --n 50

# tool hang (sim)
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_tool_hanging_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_benchmark_playback/playback_tool_hang_se.mp4 --n 100 --camera_names sideview


# observation space videos
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_obs/obs_can.mp4 --n 20 --camera_names agentview robot0_eye_in_hand
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_obs/obs_transport.mp4 --n 20 --camera_names shouldercamera0 shouldercamera1 robot0_eye_in_hand robot1_eye_in_hand
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_tool_hanging_single_expert/state_done_2.hdf5 --video_path ~/Downloads/rt_obs/obs_tool_hang.mp4 --n 20 --camera_names sideview robot0_eye_in_hand


# real videos
python playback_dataset.py --dataset ~/Desktop/real_robot/lift/demo.hdf5 --use-obs --video_path ~/Downloads/rt_benchmark_playback/playback_lift_real.mp4 --video_skip 1 --n 50
python playback_dataset.py --dataset ~/Desktop/real_robot/can/demo.hdf5 --use-obs --video_path ~/Downloads/rt_benchmark_playback/playback_can_real.mp4 --video_skip 1 --n 50
python playback_dataset.py --dataset ~/Desktop/real_robot/tool_hanging_deadline/demo.hdf5 --use-obs --video_path ~/Downloads/rt_benchmark_playback/playback_tool_hang_real.mp4 --video_skip 1 --n 50


# get first frame videos for all tasks
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_lift_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_lift_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_can_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_can_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_square_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_square_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_two_arm_transport_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_transport_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/benchmark_datasets/panda_tool_hanging_single_expert/state_done_2.hdf5 --video_path ~/Downloads/playback_first/playback_tool_hang_first.mp4 --camera_names sideview --first

### TODO: make sure to edit the file to only grab the relevant camera image first in playback_obs ###
python playback_dataset.py --dataset ~/Desktop/real_robot/lift/demo.hdf5 --use-obs --video_path ~/Downloads/playback_first/playback_lift_real_first.mp4 --first
python playback_dataset.py --dataset ~/Desktop/real_robot/can/demo.hdf5 --use-obs --video_path ~/Downloads/playback_first/playback_can_real_first.mp4 --first 
python playback_dataset.py --dataset ~/Desktop/real_robot/tool_hanging_deadline/demo.hdf5 --use-obs --video_path ~/Downloads/playback_first/playback_tool_hang_real_first.mp4 --first

# can paired BCQ rollouts
python test.py --agent /afs/cs.stanford.edu/u/amandlek/installed_libraries/benchmark/slurm/log/batchRL/batchrl_benchmark/hp_sweep/low_dim/hbcq/can_paired_bcq_low_dim/benchmark_can_paired_ld_bc_trained_models/hbcq_ds_can_paired_ld_seed_1/2021-06-12-15-12-19-179527/models/model_epoch_800_PickPlaceCan_success_0.46.pth \
--render_video --video_dir ~/Downloads/can_paired_rollouts --n_rollouts 50 --horizon 400 --seed 1

python test.py --agent /afs/cs.stanford.edu/u/amandlek/installed_libraries/benchmark/slurm/log/batchRL/batchrl_benchmark/hp_sweep/low_dim/bc/can_paired_bc_rnn_low_dim/benchmark_can_paired_ld_bc_trained_models/bc_rnn_ds_can_paired_ld_seed_1/2021-06-08-13-01-35-323829/models/model_epoch_300_PickPlaceCan_success_0.74.pth \
--render_video --video_dir ~/Downloads/can_paired_rollouts_bc_rnn --n_rollouts 50 --horizon 400 --seed 1

