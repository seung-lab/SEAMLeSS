import torch

import numpy as np

def get_black_mask(img, black_threshold):
    if black_threshold == 0:
        black_mask = img == 0
    else:
        black_mask = img <= black_threshold
    return black_mask

def get_black_fraction(img, black_threshold):
    img_black_px = get_black_mask(img, black_threshold)
    img_black_px_count = torch.sum(img_black_px)
    img_px_count = torch.sum(torch.ones_like(img))
    img_black_fraction = float(img_black_px_count) / float(img_px_count)

    return img_black_fraction

def normalize(img, per_feature_center=True, per_feature_var=False, eps=1e-5,
        mask=None, mask_fill=None):
    img_out = img.clone()
    #with torch.no_grad():
    if mask is not None:
        assert mask.shape == img.shape
    for i in range(1):
        for b in range(img.shape[0]):
            x = img_out[b]
            if per_feature_center and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        x[f][m] = x[f][m].clone() - torch.mean(x[f][m].clone())
                    else:
                        x[f] = x[f].clone() - torch.mean(x[f].clone())
            else:
                if mask is not None:
                    m = mask[b]
                    x[m] = x[m].clone() - torch.mean(x[m].clone())
                else:
                    x[...] = x.clone() - torch.mean(x.clone())

            if per_feature_var and len(img.shape) == 4:
                for f in range(img.shape[1]):
                    if mask is not None:
                        m = mask[b, f]
                        var = torch.var(x[f][m].clone())
                        x[f][m] = x[f][m].clone() / (torch.sqrt(var) + eps)
                    else:
                        var = torch.var(x[f].clone())
                        x[f] = x[f].clone() / (torch.sqrt(var) + eps)
            else:
                if mask is not None:
                    m = mask[b]
                    var = torch.var(x[m].clone())
                    x[m] = x[m].clone() / (torch.sqrt(var) + eps)
                else:
                    var = torch.var(x.clone())
                    x[...] = x.clone() / (torch.sqrt(var) + eps)

    if mask is not None and mask_fill is not None:
        img_out[mask == False] = mask_fill

    return img_out

def block_match(tgt, src, tile_size=16, tile_step=16, max_disp=10):
    src = src.squeeze()
    tgt = tgt.squeeze()

    tile_alignment_pad = (tile_size - tile_step) // 2

    padded_tgt = torch.nn.ZeroPad2d(tile_alignment_pad)(tgt)
    padded_src = torch.nn.ZeroPad2d(tile_alignment_pad)(src)

    max_disp_pad = max_disp
    padded_tgt = torch.nn.ZeroPad2d(max_disp_pad)(tgt)
    padded_src = torch.nn.ZeroPad2d(max_disp_pad)(src)

    img_size = padded_tgt.shape[-1]
    tile_count = 1 + (img_size - max_disp*2 - tile_size) // tile_step
    result = np.zeros((tile_count, tile_count, 2))

    for x_tile in range(0, tile_count):
        for y_tile in range(0, tile_count):
            src_tile_coord, tgt_tile_coord = compute_tile_coords(x_tile, y_tile, tile_size,
                                                              tile_step, max_disp, img_size,
                                                                x_offset=max_disp,
                                                                y_offset=max_disp)
            src_tile = padded_src[src_tile_coord]
            tgt_tile = padded_tgt[tgt_tile_coord]
            if get_black_fraction(src_tile, 0) > 0.7 or get_black_fraction(tgt_tile, 0) > 0.95:
                match_displacement = [0, 0]
                #print ('skipping: {} {}'.format(get_black_fraction(src_tile, 0), get_black_fraction(tgt_tile, 0)))
            else:
                ncc = get_ncc(tgt_tile, src_tile)
                match = np.unravel_index(ncc.argmax(), ncc.shape)

                match_tile_start = (tgt_tile_coord[0].start + match[0], tgt_tile_coord[1].start + match[1])
                src_tile_start   = (src_tile_coord[0].start, src_tile_coord[1].start)
                match_displacement = np.subtract(src_tile_start, match_tile_start)

            result[x_tile, y_tile, 0] = -match_displacement[1]
            result[x_tile, y_tile, 1] = -match_displacement[0]

    result_var = torch.cuda.FloatTensor(result).unsqueeze(0)
    scale = tgt.shape[-2] / result_var.shape[-2]
    result_ups_var = torch.nn.functional.interpolate(result_var.permute(0, 3, 1, 2),
            scale_factor=scale, mode='bilinear')

    final_result_var = result_ups_var
    final_result = final_result_var.permute(0, 2, 3, 1)
    return final_result

def get_ncc(tgt, tmpl):
    tgt = tgt.unsqueeze(0).unsqueeze(0)
    tmpl = tmpl.unsqueeze(0).unsqueeze(0)
    #import pdb; pdb.set_trace()
    tgt_norm = normalize(tgt, mask=(tgt != 0))
    tmpl_norm = normalize(tmpl, mask=(tmpl != 0))
    ncc = get_cc(tgt_norm, tmpl_norm)
    return ncc.squeeze().cpu().numpy()

def get_cc(target, template, feature_weights=None, normalize_cc=False):
    cc_side = target.shape[-1] - template.shape[-1] + 1
    cc = torch.zeros((target.shape[0], cc_side, cc_side),
            device=target.device, dtype=torch.float)
    for b in range(target.shape[0]):
        cc[b:b+1] = torch.nn.functional.conv2d(target[b:b+1], template[b:b+1]).squeeze()
    #print (template.shape)
    if normalize_cc:
        cc = normalize(cc)
    else:
        cc = cc / torch.sum(torch.ones_like(template[0], device=template.device))
    #print (torch.mean(cc))
    return cc

def get_displaced_tile(disp, tile):
    result = copy.deepcopy(tile)
    result[0].start += disp[0]

def compute_tile_coords(x_tile, y_tile, tile_size, tile_step, max_disp, img_size,
                       x_offset=0, y_offset=0):
    #import pdb; pdb.set_trace()
    src_xs = x_tile * tile_step + x_offset
    src_xe = src_xs + tile_size
    src_ys = y_tile * tile_step + y_offset
    src_ye = src_ys + tile_size

    tgt_xs = max(0, src_xs - max_disp)
    tgt_xe = min(img_size, src_xe + max_disp)
    tgt_ys = max(0, src_ys - max_disp)
    tgt_ye = min(img_size, src_ye + max_disp)

    src_coords = (slice(src_xs, src_xe), slice(src_ys, src_ye))
    tgt_coords = (slice(tgt_xs, tgt_xe), slice(tgt_ys, tgt_ye))

    return src_coords, tgt_coords

def get_patch_middle(coords):
    x_m = (coords[0].start - coords[0].end) / 2
    y_m = (coords[1].start - coords[1].end) / 2

    return(x_m, y_m)
