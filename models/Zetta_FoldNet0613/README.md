# Zetta FoldNet0613

Trained by Alexander Bae (jabae@princeton.edu)  
  
exp dir: 'ZettaAI/Kubota/FoldDetection/groundtruth/train/train_mip2_320_0613'  
chkpt num: 122k  
net architecture: detect_net  
mip: 2 (20 nm x 20 nm)  
conv mode: valid  
patch size: 320 x 320  
eff. patch size: 256 x 256  
overlap: 32  
train data: FAFB  
augmentation: flip, rotate90, contrast, blackpad  
pretrain: FAFB_FoldNet0515 (finetuned)  
