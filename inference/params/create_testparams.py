
for model_name in ['zzzz_02_16_mip_4_6_8_fold_xy130k_badcoarse_v3_finetune_sm1e7', 'zzzz_02_16_mip_4_6_8_fold_xy130k_badcoarse_v3_finetune_sm2e7',
        'zzzz_02_16_mip_4_6_8_fold_xy130k_badcoarse_v3_finetune_sm3e7']:
    with open('./sergiy_test_template.csv', 'r') as f_in:
        with open('./sergiy_test_{}.csv'.format(model_name), 'w') as f_out:
            l = f_in.read()
            f_out.write(l.replace('SERGIYMODEL', 'sergiy_m4m6m8_' + model_name))
