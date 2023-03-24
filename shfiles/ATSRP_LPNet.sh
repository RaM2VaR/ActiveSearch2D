# # #!/bin/bash

# # # file_list=['/home/ubuntu/a.py','/home/ubuntu/b.py','/home/ubuntu/c.py']

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.001' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.002' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.004' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.006' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.008' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.01' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.001' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.002' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.004' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.006' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.008' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_16_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.01' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.001' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.002' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.004' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.006' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.008' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_1.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.01' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.001' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.002' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.004' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.006' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.008' 
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'active_search.py' 'graphs_for_publication/random_TGFF_64_2.pt' '--lr' '0.001' '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.01' 
done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'active_search.py' 'traffic_benchmark/vopd_norm.pt' '--lr' '0.001'  '--pretrained_model_path' 'models_data_final/model_pretrain_smallData_04-24_17-37.pt'   '--max_iter' '1000' '--num_samples' '1024' '--obj_fun' 'LPNet' '--inj_rate' '0.001'
# done