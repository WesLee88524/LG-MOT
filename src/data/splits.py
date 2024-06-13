import os.path as osp


def get_seqs_from_splits(data_path, train_split=None, val_split=None, test_split=None):
    """
    Get splits that will be used in the experiment
    """
    _SPLITS = {}

    # MOT17 Dets
    mot17_dets = ('SDP', 'FRCNN', 'DPM')


    # Full MOT17-Train
    _SPLITS['mot17-train-all'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 5, 9, 10, 11, 13) for det in mot17_dets]}
    _SPLITS['mot17-train-split1'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 5, 9, 11) for det in mot17_dets]}
    _SPLITS['mot17-val-split1'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 10, 13) for det in mot17_dets]}
    _SPLITS['mot17-train-split2'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 5, 9, 10, 13) for det in mot17_dets]}
    _SPLITS['mot17-val-split2'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (4, 11) for det in mot17_dets]}
    _SPLITS['mot17-train-split3'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (2, 4, 10, 11, 13) for det in mot17_dets]}
    _SPLITS['mot17-val-split3'] = {'MOT17/train': [f'MOT17-{seq_num:02}-{det}' for seq_num in (5, 9) for det in mot17_dets]}

    # MOT 17 test set
    _SPLITS['mot17-test-all'] = {
        'MOT17/test': [f'MOT17-{seq_num:02}-{det}' for seq_num in (1, 3, 6, 7, 8, 12, 14) for det in mot17_dets]}

    #######
    # MOT20
    #######
    _SPLITS['mot20-train-all'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 3, 5)]}
    _SPLITS['mot20-test-all'] = {'MOT20/test': [f'MOT20-{seq_num:02}' for seq_num in (4, 6, 7, 8)]}

    _SPLITS['mot20-train-split1'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 3)]}
    _SPLITS['mot20-train-split2'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 2, 5)]}
    _SPLITS['mot20-train-split3'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1, 3, 5)]}
    _SPLITS['mot20-train-split4'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (2, 3, 5)]}

    _SPLITS['mot20-val-split1'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (5,)]}
    _SPLITS['mot20-val-split2'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (3,)]}
    _SPLITS['mot20-val-split3'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (2,)]}
    _SPLITS['mot20-val-split4'] = {'MOT20/train': [f'MOT20-{seq_num:02}' for seq_num in (1,)]}



    #######
    # DanceTrack
    #######
    dancetrack_train_seqs = (1, 2, 6, 8, 12, 15, 16, 20, 23, 24, 27, 29, 32, 33, 37, 39, 44, 45, 49, 51, 52, 53, 55, 57, 61, 62, 66, 68, 69, 72, 74, 75, 80, 82, 83, 86, 87, 96, 98, 99)
    dancetrack_val_seqs = (4, 5, 7, 10, 14, 18, 19, 25, 26, 30, 34, 35, 41, 43, 47, 58, 63, 65, 73, 77, 79, 81, 90, 94, 97)
    dancetrack_test_seqs = (21, 3, 9, 11, 13, 17, 22, 28, 31, 36, 38, 40, 42, 46, 48, 50, 54, 56, 59, 60, 64, 67, 70, 71, 76, 78, 84, 85, 88, 89, 91, 92, 93, 95, 100)

    assert set(dancetrack_train_seqs + dancetrack_val_seqs + dancetrack_test_seqs) == set([x for x in range(1, 101)]), "Missing sequence in the dancetrack splits"
    assert len(dancetrack_train_seqs + dancetrack_val_seqs + dancetrack_test_seqs) == 100, "Missing or duplicate sequence in the dancetrack splits"

    _SPLITS['dancetrack-train-all'] = {'DANCETRACK/train': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_train_seqs]}
    _SPLITS['dancetrack-val-all'] = {'DANCETRACK/val': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_val_seqs]}
    _SPLITS['dancetrack-test-all'] = {'DANCETRACK/test': [f'dancetrack{seq_num:04}' for seq_num in dancetrack_test_seqs]}

    _SPLITS['dancetrack-debug'] = {'DANCETRACK/train': [f'dancetrack{seq_num:04}' for seq_num in (1,)]}
    _SPLITS['dancetrack-val-debug'] = {'DANCETRACK/val': [f'dancetrack{seq_num:04}' for seq_num in (4,)]}

    #######
    # SportsMOT
    #######
    _SPLITS['SportsMOT-train-all'] = {'SPORTSMOT/train': ['v_-6Os86HzwCs_c001', 'v_-6Os86HzwCs_c003', 'v_-6Os86HzwCs_c007', 'v_-6Os86HzwCs_c009', 'v_1LwtoLPw2TU_c006', 'v_1LwtoLPw2TU_c012', 'v_1LwtoLPw2TU_c014', 'v_1LwtoLPw2TU_c016', 'v_1yHWGw8DH4A_c029', 'v_1yHWGw8DH4A_c047', 'v_1yHWGw8DH4A_c077', 'v_1yHWGw8DH4A_c601', 'v_1yHWGw8DH4A_c609', 'v_1yHWGw8DH4A_c610', 'v_2j7kLB-vEEk_c001', 'v_2j7kLB-vEEk_c002', 'v_2j7kLB-vEEk_c005', 'v_2j7kLB-vEEk_c007', 'v_2j7kLB-vEEk_c009', 'v_2j7kLB-vEEk_c010', 'v_4LXTUim5anY_c002', 'v_4LXTUim5anY_c003', 'v_4LXTUim5anY_c010', 'v_4LXTUim5anY_c012', 'v_4LXTUim5anY_c013', 'v_ApPxnw_Jffg_c001', 'v_ApPxnw_Jffg_c002', 'v_ApPxnw_Jffg_c009', 'v_ApPxnw_Jffg_c015', 'v_ApPxnw_Jffg_c016', 'v_CW0mQbgYIF4_c004', 'v_CW0mQbgYIF4_c005', 'v_CW0mQbgYIF4_c006', 'v_Dk3EpDDa3o0_c002', 'v_Dk3EpDDa3o0_c007', 'v_HdiyOtliFiw_c003', 'v_HdiyOtliFiw_c004', 'v_HdiyOtliFiw_c008', 'v_HdiyOtliFiw_c010', 'v_HdiyOtliFiw_c602', 'v_dChHNGIfm4Y_c003', 'v_gQNyhv8y0QY_c003', 'v_gQNyhv8y0QY_c012', 'v_gQNyhv8y0QY_c013', 'v_iIxMOsCGH58_c013']}
    _SPLITS['SportsMOT-val-all'] = {'SPORTSMOT/val': ['v_00HRwkvvjtQ_c001', 'v_00HRwkvvjtQ_c003', 'v_00HRwkvvjtQ_c005', 'v_00HRwkvvjtQ_c007', 'v_00HRwkvvjtQ_c008', 'v_00HRwkvvjtQ_c011', 'v_0kUtTtmLaJA_c004', 'v_0kUtTtmLaJA_c005', 'v_0kUtTtmLaJA_c006', 'v_0kUtTtmLaJA_c007', 'v_0kUtTtmLaJA_c008', 'v_0kUtTtmLaJA_c010', 'v_2QhNRucNC7E_c017', 'v_4-EmEtrturE_c009', 'v_4r8QL_wglzQ_c001', 'v_5ekaksddqrc_c001', 'v_5ekaksddqrc_c002', 'v_5ekaksddqrc_c003', 'v_5ekaksddqrc_c004', 'v_5ekaksddqrc_c005', 'v_9MHDmAMxO5I_c002', 'v_9MHDmAMxO5I_c003', 'v_9MHDmAMxO5I_c004', 'v_9MHDmAMxO5I_c006', 'v_9MHDmAMxO5I_c009', 'v_BgwzTUxJaeU_c008', 'v_BgwzTUxJaeU_c012', 'v_BgwzTUxJaeU_c014', 'v_G-vNjfx1GGc_c004', 'v_G-vNjfx1GGc_c008', 'v_G-vNjfx1GGc_c600', 'v_G-vNjfx1GGc_c601', 'v_ITo3sCnpw_k_c007', 'v_ITo3sCnpw_k_c010', 'v_ITo3sCnpw_k_c011', 'v_ITo3sCnpw_k_c012', 'v_cC2mHWqMcjk_c007', 'v_cC2mHWqMcjk_c008', 'v_cC2mHWqMcjk_c009', 'v_dw7LOz17Omg_c053', 'v_dw7LOz17Omg_c067', 'v_i2_L4qquVg0_c006', 'v_i2_L4qquVg0_c007', 'v_i2_L4qquVg0_c009', 'v_i2_L4qquVg0_c010']}
    _SPLITS['SportsMOT-test-all'] = {'SPORTSMOT/test': ['v_-9kabh1K8UA_c008', 'v_-9kabh1K8UA_c009', 'v_-9kabh1K8UA_c010', 'v_-hhDbvY5aAM_c001', 'v_-hhDbvY5aAM_c002', 'v_-hhDbvY5aAM_c005', 'v_-hhDbvY5aAM_c006', 'v_-hhDbvY5aAM_c007', 'v_-hhDbvY5aAM_c008', 'v_-hhDbvY5aAM_c009', 'v_-hhDbvY5aAM_c011', 'v_-hhDbvY5aAM_c012', 'v_-hhDbvY5aAM_c600', 'v_-hhDbvY5aAM_c602', 'v_1UDUODIBSsc_c001', 'v_1UDUODIBSsc_c004', 'v_1UDUODIBSsc_c015', 'v_1UDUODIBSsc_c037', 'v_1UDUODIBSsc_c055', 'v_1UDUODIBSsc_c056', 'v_1UDUODIBSsc_c063', 'v_1UDUODIBSsc_c067', 'v_1UDUODIBSsc_c090', 'v_1UDUODIBSsc_c103', 'v_1UDUODIBSsc_c111', 'v_1UDUODIBSsc_c113', 'v_1UDUODIBSsc_c601', 'v_1UDUODIBSsc_c602', 'v_1UDUODIBSsc_c603', 'v_1UDUODIBSsc_c606', 'v_1UDUODIBSsc_c609', 'v_1UDUODIBSsc_c615', 'v_2BhBRkkAqbQ_c002', 'v_2ChiYdg5bxI_c039', 'v_2ChiYdg5bxI_c044', 'v_2ChiYdg5bxI_c046', 'v_2ChiYdg5bxI_c058', 'v_2ChiYdg5bxI_c067', 'v_2ChiYdg5bxI_c072', 'v_2ChiYdg5bxI_c086', 'v_2ChiYdg5bxI_c115', 'v_2ChiYdg5bxI_c120', 'v_2ChiYdg5bxI_c128', 'v_2ChiYdg5bxI_c135', 'v_2ChiYdg5bxI_c136', 'v_2ChiYdg5bxI_c600', 'v_2ChiYdg5bxI_c601', 'v_2ChiYdg5bxI_c602', 'v_2Dnx8BpgUEs_c003', 'v_2Dnx8BpgUEs_c005', 'v_2Dnx8BpgUEs_c006', 'v_2Dnx8BpgUEs_c007', 'v_2Dw9QNH5KtU_c001', 'v_2Dw9QNH5KtU_c003', 'v_2Dw9QNH5KtU_c004', 'v_2Dw9QNH5KtU_c007', 'v_2Dw9QNH5KtU_c008', 'v_2Dw9QNH5KtU_c012', 'v_2Dw9QNH5KtU_c014', 'v_2Dw9QNH5KtU_c018', 'v_42VrMbd68Zg_c003', 'v_42VrMbd68Zg_c004', 'v_42VrMbd68Zg_c011', 'v_42VrMbd68Zg_c012', 'v_6OLC1-bhioc_c001', 'v_6OLC1-bhioc_c003', 'v_6OLC1-bhioc_c005', 'v_6OLC1-bhioc_c007', 'v_6OLC1-bhioc_c008', 'v_6OLC1-bhioc_c009', 'v_6OLC1-bhioc_c011', 'v_6OLC1-bhioc_c012', 'v_6OLC1-bhioc_c013', 'v_6OLC1-bhioc_c014', 'v_6oTxfzKdG6Q_c004', 'v_6oTxfzKdG6Q_c015', 'v_7FTsO8S3h88_c001', 'v_7FTsO8S3h88_c004', 'v_7FTsO8S3h88_c005', 'v_7FTsO8S3h88_c006', 'v_7FTsO8S3h88_c007', 'v_7FTsO8S3h88_c008', 'v_82WN8C8qJmI_c002', 'v_82WN8C8qJmI_c003', 'v_82WN8C8qJmI_c004', 'v_82WN8C8qJmI_c005', 'v_82WN8C8qJmI_c014', 'v_8rG1vjmJHr4_c004', 'v_8rG1vjmJHr4_c005', 'v_8rG1vjmJHr4_c006', 'v_8rG1vjmJHr4_c007', 'v_8rG1vjmJHr4_c008', 'v_8rG1vjmJHr4_c010', 'v_9p0i81kAEwE_c002', 'v_9p0i81kAEwE_c003', 'v_9p0i81kAEwE_c006', 'v_9p0i81kAEwE_c007', 'v_9p0i81kAEwE_c008', 'v_9p0i81kAEwE_c009', 'v_9p0i81kAEwE_c010', 'v_9p0i81kAEwE_c013', 'v_9p0i81kAEwE_c014', 'v_9p0i81kAEwE_c015', 'v_9p0i81kAEwE_c018', 'v_9p0i81kAEwE_c019', 'v_A4OJhlI6hgc_c001', 'v_A4OJhlI6hgc_c002', 'v_A4OJhlI6hgc_c003', 'v_A4OJhlI6hgc_c005', 'v_A4OJhlI6hgc_c601', 'v_BdD9xu0E2H4_c003', 'v_BdD9xu0E2H4_c004', 'v_BdD9xu0E2H4_c006', 'v_BdD9xu0E2H4_c010', 'v_BdD9xu0E2H4_c011', 'v_BdD9xu0E2H4_c013', 'v_DjtFlW2eHFI_c614', 'v_DjtFlW2eHFI_c616', 'v_aAb0psypDj4_c003', 'v_aAb0psypDj4_c006', 'v_aAb0psypDj4_c008', 'v_aAb0psypDj4_c009', 'v_aVfJwdQxCsU_c001', 'v_aVfJwdQxCsU_c007', 'v_aVfJwdQxCsU_c015', 'v_bQNDRprvpus_c001', 'v_bQNDRprvpus_c003', 'v_bQNDRprvpus_c005', 'v_bQNDRprvpus_c007', 'v_bQNDRprvpus_c008', 'v_bhWjlAEICp8_c003', 'v_bhWjlAEICp8_c004', 'v_bhWjlAEICp8_c006', 'v_bhWjlAEICp8_c008', 'v_bhWjlAEICp8_c009', 'v_bhWjlAEICp8_c010', 'v_bhWjlAEICp8_c014', 'v_bhWjlAEICp8_c018', 'v_czYZnO9QxYQ_c002', 'v_czYZnO9QxYQ_c005', 'v_czYZnO9QxYQ_c008', 'v_czYZnO9QxYQ_c012', 'v_czYZnO9QxYQ_c013', 'v_czYZnO9QxYQ_c014', 'v_czYZnO9QxYQ_c016', 'v_czYZnO9QxYQ_c020', 'v_foGDoBblREI_c002', 'v_foGDoBblREI_c008', 'v_iF9bKPWdZlc_c001', 'v_iF9bKPWdZlc_c003']}
    _SPLITS['SportsMOT-trainval-all'] = {'SPORTSMOT/trainval': ['v_-6Os86HzwCs_c001', 'v_-6Os86HzwCs_c003', 'v_-6Os86HzwCs_c007', 'v_-6Os86HzwCs_c009', 'v_1LwtoLPw2TU_c006', 'v_1LwtoLPw2TU_c012', 'v_1LwtoLPw2TU_c014', 'v_1LwtoLPw2TU_c016', 'v_1yHWGw8DH4A_c029', 'v_1yHWGw8DH4A_c047', 'v_1yHWGw8DH4A_c077', 'v_1yHWGw8DH4A_c601', 'v_1yHWGw8DH4A_c609', 'v_1yHWGw8DH4A_c610', 'v_2j7kLB-vEEk_c001', 'v_2j7kLB-vEEk_c002', 'v_2j7kLB-vEEk_c005', 'v_2j7kLB-vEEk_c007', 'v_2j7kLB-vEEk_c009', 'v_2j7kLB-vEEk_c010', 'v_4LXTUim5anY_c002', 'v_4LXTUim5anY_c003', 'v_4LXTUim5anY_c010', 'v_4LXTUim5anY_c012', 'v_4LXTUim5anY_c013', 'v_ApPxnw_Jffg_c001', 'v_ApPxnw_Jffg_c002', 'v_ApPxnw_Jffg_c009', 'v_ApPxnw_Jffg_c015', 'v_ApPxnw_Jffg_c016', 'v_CW0mQbgYIF4_c004', 'v_CW0mQbgYIF4_c005', 'v_CW0mQbgYIF4_c006', 'v_Dk3EpDDa3o0_c002', 'v_Dk3EpDDa3o0_c007', 'v_HdiyOtliFiw_c003', 'v_HdiyOtliFiw_c004', 'v_HdiyOtliFiw_c008', 'v_HdiyOtliFiw_c010', 'v_HdiyOtliFiw_c602', 'v_dChHNGIfm4Y_c003', 'v_gQNyhv8y0QY_c003', 'v_gQNyhv8y0QY_c012', 'v_gQNyhv8y0QY_c013', 'v_iIxMOsCGH58_c013','v_00HRwkvvjtQ_c001', 'v_00HRwkvvjtQ_c003', 'v_00HRwkvvjtQ_c005', 'v_00HRwkvvjtQ_c007', 'v_00HRwkvvjtQ_c008', 'v_00HRwkvvjtQ_c011', 'v_0kUtTtmLaJA_c004', 'v_0kUtTtmLaJA_c005', 'v_0kUtTtmLaJA_c006', 'v_0kUtTtmLaJA_c007', 'v_0kUtTtmLaJA_c008', 'v_0kUtTtmLaJA_c010', 'v_2QhNRucNC7E_c017', 'v_4-EmEtrturE_c009', 'v_4r8QL_wglzQ_c001', 'v_5ekaksddqrc_c001', 'v_5ekaksddqrc_c002', 'v_5ekaksddqrc_c003', 'v_5ekaksddqrc_c004', 'v_5ekaksddqrc_c005', 'v_9MHDmAMxO5I_c002', 'v_9MHDmAMxO5I_c003', 'v_9MHDmAMxO5I_c004', 'v_9MHDmAMxO5I_c006', 'v_9MHDmAMxO5I_c009', 'v_BgwzTUxJaeU_c008', 'v_BgwzTUxJaeU_c012', 'v_BgwzTUxJaeU_c014', 'v_G-vNjfx1GGc_c004', 'v_G-vNjfx1GGc_c008', 'v_G-vNjfx1GGc_c600', 'v_G-vNjfx1GGc_c601', 'v_ITo3sCnpw_k_c007', 'v_ITo3sCnpw_k_c010', 'v_ITo3sCnpw_k_c011', 'v_ITo3sCnpw_k_c012', 'v_cC2mHWqMcjk_c007', 'v_cC2mHWqMcjk_c008', 'v_cC2mHWqMcjk_c009', 'v_dw7LOz17Omg_c053', 'v_dw7LOz17Omg_c067', 'v_i2_L4qquVg0_c006', 'v_i2_L4qquVg0_c007', 'v_i2_L4qquVg0_c009', 'v_i2_L4qquVg0_c010']}
    ########
    # BDD
    ########
    _SPLITS['bdd-val-debug'] = {'BDD/val': [f'{seq_name}' for seq_name in ('b1c66a42-6f7d68ca', 'b1c9c847-3bda4659')]}


    # Ensure that split is valid
    assert train_split in _SPLITS.keys() or train_split is None, "Training split is not valid!"
    assert val_split in _SPLITS.keys() or val_split is None, "Validation split is not valid!"
    assert test_split in _SPLITS.keys() or test_split is None, "Test split is not valid!"

    # Get the sequences to use in the experiment
    seqs = {}
    if train_split is not None:
        seqs['train'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                         _SPLITS[train_split].items()}
    if val_split is not None:
        seqs['val'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                       _SPLITS[val_split].items()}
    if test_split is not None:
        seqs['test'] = {osp.join(data_path, seqs_path): seq_list for seqs_path, seq_list in
                        _SPLITS[test_split].items()}
    return seqs, (train_split, val_split, test_split)
