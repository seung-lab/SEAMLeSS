def get_param_set(params):
    max_mip = 9
    if params == 0:
        # basil test
        source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
        v_off = (102400, 102400, 15)
        x_size = 10240*4
        y_size = 10240*4
    elif params == 1:
        # basil folds
        #source = 'gs://neuroglancer/basil_v0/father_of_alignment/v3'
        #source = 'gs://neuroglancer/nflow_tests/bprodsmooth_crack_pass/image'
        v_off = (10240*18, 10240*4, 179)
        x_size = 1024*16
        y_size = 1024*16
    elif params == 2:
        # fly normal
        source = 'gs://neuroglancer/drosophila_v0/image_v14_single_slices'
        v_off = (10240*12, 10240 * 4, 2410)
        x_size = 10240 * 4
        y_size = 10240 * 4
        max_mip = 6
    elif params == 3:
        # basil big
        source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
        v_off = (10240*4, 10240*2, 179)
        x_size = 10240*16
        y_size = 10240*16
    elif params == 4:
        # basil edge
        source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
        v_off = (1024*8, 10240*8, 179)
        x_size = 10240*4
        y_size = 10240*4
    elif params == 5:
        # basil corner
        source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
        v_off = (1024*8, 1024*10, 179)
        x_size = 10240*4
    elif params == 6:
        # basil corner defect
        source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
        v_off = (1024*8, 1024*10, 187)
        x_size = 10240*4
        y_size = 10240*4
    elif params == 7:
        # basil raw
        source = 'gs://neuroglancer/basil_v0/raw_image_cropped'
        v_off = (10240*18, 10240*3, 179)
        x_size = 10240*3
        y_size = 10240*3
    elif params == 8:
        # basil father of alignment
        source = 'gs://neuroglancer/basil_v0/father_of_alignment/v3'
        v_off = (10240*18, 10240*3, 179)
        x_size = 10240*3
        y_size = 10240*3
    elif params == 9:
        # basil son of alignment
        source = 'gs://neuroglancer/seamless/cprod46_correct_enc_side_father/image'
        v_off = (10240*18, 10240*3, 179)
        x_size = 10240*3
        y_size = 10240*3
    elif params == 10:
        # basil son of alignment
        source = 'gs://neuroglancer/seamless/cprod_defect_net5_side_father/image'
        v_off = (10240*18, 10240*3, 179)
        x_size = 10240*3
        y_size = 10240*3
    else:
        raise Exception('Invalid param set for inference: {}.'.format(index))

    return source, v_off, x_size, y_size, max_mip
