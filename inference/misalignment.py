def misalignment_detector(img1, img2, mip):
    return (img1 - img2).abs()
