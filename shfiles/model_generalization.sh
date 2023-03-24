for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF1_norm_16.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG16_1.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF2_norm_16.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG16_2.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF1_norm_25.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG25_1.txt
done


for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF2_norm_25.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG25_2.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF1_norm_36.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG36_1.txt
done


for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF2_norm_36.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG36_2.txt
done


for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF1_norm_49.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG49_1.txt
done


for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF2_norm_49.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG49_2.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF1_norm_64.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG81_1.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF2_norm_64.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG81_2.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF1_norm_81.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG81_1.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF2_norm_81.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '512' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG81_2.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF1_norm_121.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG121_1.txt
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'data_TGFF/data_single_TGFF2_norm_121.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_TG121_2.txt
done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/263DEC.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_263DEC.txt
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/263ENC_norm.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_263ENC.txt
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/MP3ENC.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_MP3ENC.txt
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/MPEG-4_norm.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_MPEG-4.txt
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/MWD_norm.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_MWD.txt
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/PIP_norm.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_PIP.txt
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/vopd_norm.pt' '--lr' '0.0001'    '--max_iter' '100' '--num_samples' '256' '--obj_fun' 'comm_cost' '--inj_rate' '0.001' '--pretrained_model_path' 'model_gen_for_revision81.pt' >> results_for_revision/genrl_notrain81_vopd.txt
# done

