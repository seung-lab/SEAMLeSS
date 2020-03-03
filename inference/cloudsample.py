from fields import Field

def cloudsample_compose(f_cv, g_cv, f_z, g_z, bbox, f_mip, g_mip,
                        dst_mip, factor=1., pad=256):
    """Wrapper for torch.nn.functional.gridsample for CloudVolume field objects.

    Gridsampling a field is a composition, such that f(g(x)).

    Args:
       f_cv: MiplessCloudVolume storing the vector field to do the warping
       g_cv: MiplessCloudVolume storing the vector field to be warped
       bbox: BoundingBox for output region to be warped
       f_z, g_z: int for section index from which to read fields
       f_mip, g_mip: int for MIPs of the input fields
       dst_mip: int for MIP of the desired output field
       factor: float to multiply the f vector field by
       pad: number of pixels to pad at dst_mip

    Returns:
       composed field
    """
    return cloudsample_multicompose(field_list=[f_cv, g_cv], 
                             z_list=[f_z, g_z], 
                             bbox=bbox, 
                             mip_list=[f_mip, g_mip],
                             dst_mip=dst_mip,
                             factors=[factor, factor],
                             pad=pad)

def cloudsample_multicompose(field_list, z_list, bbox, mip_list,
                              dst_mip, factors=None, pad=256):
    """Compose a list of FieldCloudVolumes
  
    This takes a list of fields
    field_list = [f_0, f_1, ..., f_n]
    and composes them to get
    f_0 ⚬ f_1 ⚬ ... ⚬ f_n ~= f_0(f_1(...(f_n)))
  
    Args:
       field_list: list of MiplessCloudVolume storing the vector fields
       z_list: int or list of ints for section indices to read fields
       bbox: BoundingBox for output region to be warped
       mip_list: int or list of ints for MIPs of the input fields
       dst_mip: int for MIP of the desired output field
       factors: floats to multiply/reweight the fields by before composing
       pad: number of pixels to pad at dst_mip
  
    Returns:
       composed field
    """
    if isinstance(z_list, int):
        z_list = [z_list] * len(field_list)
    else:
        assert(len(z_list) == len(field_list))
    if isinstance(mip_list, int):
        mip_list = [mip_list] * len(field_list)
    else:
        assert(len(mip_list) == len(field_list))
    assert(min(mip_list) >= dst_mip)
    if factors is None:
        factors = [1.0] * len(field_list)
    else:
        assert(len(factors) == len(field_list))
    padded_bbox = bbox.copy()
    padded_bbox.max_mip = dst_mip
    print('Padding by {} at MIP{}'.format(pad, dst_mip))
    slices = slice(None), slice(pad,-pad), slice(pad,-pad), slice(None)
    if pad == 0:
        slices = slice(None), slice(None), slice(None), slice(None)
    padded_bbox.uncrop(pad, mip=dst_mip)
  
    # load the first vector field
    f_cv, *field_list = field_list
    f_z, *z_list = z_list
    f_mip, *mip_list = mip_list
    f_factor, *factors = factors
    f = f_cv[f_mip][padded_bbox.to_cube(f_z)]

    f = f * f_factor
    if len(field_list) == 0:
        return Field(f.field[slices], bbox)
  
    # skip any empty / identity fields
    while f.is_identity(magn_eps=1e-6):
        f_cv, *field_list = field_list
        f_z, *z_list = z_list
        f_mip, *mip_list = mip_list
        f_factor, *factors = factors
        f = f_cv[f_mip][padded_bbox.to_cube(f_z)]
        f = f * f_factor
        if len(field_list) == 0:
            return Field(f.field[slices], bbox)
  
    if f_mip > dst_mip:
        f = f.up(f_mip - dst_mip)
  
    # compose with the remaining fields
    while len(field_list) > 0:
        g_cv, *field_list = field_list
        g_z, *z_list = z_list
        g_mip, *mip_list = mip_list
        g_factor, *factors = factors
  
        dist = f.profile(keepdim=True)
        dist = (dist // (2**g_mip)) * 2**g_mip
        new_bbox = padded_bbox.translate(dist[0,:,0,0]) 
  
        f -= dist
        g = g_cv[g_mip][new_bbox.to_cube(g_z)]
        g = g * g_factor
        if g_mip > dst_mip:
            g = g.up(g_mip - dst_mip)
        h = f(g)
        h += dist
        f = h

    return Field(f.field[slices], bbox)

