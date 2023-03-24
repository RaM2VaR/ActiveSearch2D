#!/bin/bash

# file_list=['/home/ubuntu/a.py','/home/ubuntu/b.py','/home/ubuntu/c.py']

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_16.pt' '--num_prtl' '100' '--max_iter' '256'  '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_16.pt' '--num_prtl' '100' '--max_iter' '256'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_25.pt' '--num_prtl' '100' '--max_iter' '625'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_25.pt' '--num_prtl' '100' '--max_iter' '625'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_36.pt' '--num_prtl' '100' '--max_iter' '1296'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_36.pt' '--num_prtl' '100' '--max_iter' '1296'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_49.pt' '--num_prtl' '100' '--max_iter' '2000'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_49.pt' '--num_prtl' '100' '--max_iter' '2000'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_64.pt' '--num_prtl' '100' '--max_iter' '2000'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_64.pt' '--num_prtl' '100' '--max_iter' '2000'    '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_81.pt' '--num_prtl' '100' '--max_iter' '2000'  '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_81.pt' '--num_prtl' '100' '--max_iter' '2000' '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_100.pt' '--num_prtl' '100' '--max_iter' '2000'  '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_100.pt' '--num_prtl' '100' '--max_iter' '2000' '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF1_norm_121.pt' '--num_prtl' '100' '--max_iter' '2000'  '--obj_fun' 'comm_energy'   
done

for i  in 1 2 3 4 5
do
    # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
    python 'ga_tosun_new.py' 'data_TGFF/data_single_TGFF2_norm_121.pt' '--num_prtl' '100' '--max_iter' '2000' '--obj_fun' 'comm_energy'   
done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'ga_tosun_new.py' 'traffic_benchmark/vopd_norm.pt' '--num_prtl' '100' '--max_iter' '256' '--obj_fun' 'comm_energy'   
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'ga_tosun_new.py' 'traffic_benchmark/MWD_norm.pt' '--num_prtl' '100' '--max_iter' '144' '--obj_fun' 'comm_energy'   
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'ga_tosun_new.py' 'traffic_benchmark/MPEG-4_norm.pt' '--num_prtl' '100' '--max_iter' '144' '--obj_fun' 'comm_energy'   
# done

# for i  in 1 2 3 4 5
# do
#     # python '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/active_search_3D.py' '-p' '/home/ram_lak/Ramesh_work/RL_work/AS_3D_May/Pkl/mesh_442.pkl'
#     python 'ga_tosun_new.py' 'traffic_benchmark/263ENC_norm.pt' '--num_prtl' '100' '--max_iter' '144' '--obj_fun' 'comm_energy'   
# done