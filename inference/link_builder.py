
def ng_link(out_name, out_full_path, source_name, source_full_path, x_center, y_center, z_start):
    link = "https://neuromancer-seung-import.appspot.com/#!{'layers':{'__OUT_NAME__':{'type':'image'_'source':'__OUT_FULL_PATH__'}_'__SOURCE_NAME__':{'type':'image'_'source':'__SOURCE_FULL_PATH__'}}_'navigation':{'pose':{'position':{'voxelSize':[4_4_40]_'voxelCoordinates':[__X_CENTER_____Y_CENTER_____Z_START__]}}_'zoomFactor':150}_'layout':'xy'}"
    link = link.replace('__OUT_NAME__', out_name)
    link = link.replace('__OUT_FULL_PATH__', out_full_path)
    link = link.replace('__SOURCE_NAME__', source_name)
    link = link.replace('__SOURCE_FULL_PATH__', source_full_path)
    link = link.replace('__X_CENTER__', str(x_center))
    link = link.replace('__Y_CENTER__', str(y_center))
    link = link.replace('__Z_START__', str(z_start))
    return link
