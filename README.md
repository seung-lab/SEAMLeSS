# Neuroflow
Compute Flow across sections and align.

# Run

Run the environment
```
nvidia-docker run -it --net=host \
      -v $(pwd):/Neuroflow
      -v /usr/people/$USER/.neuroglancer/secrets/:/root/.cloudvolume/secrets/ \
      -e GOOGLE_APPLICATION_CREDENTIALS='/root/.cloudvolume/secrets/google-secret.json' \
      davidbun/cavelab:latest-gpu bash
```
And to train

```
cd /Neuroflow
python neuroflow/train.py
```

# Goals
1. small dataset (Proof of concept)
2. a section (MVP)
3. Pinky 10 sections (beta)
4. Pinky 100 sections (alpha)
5. Phase II (production)

# Todos
- Training
  V data (in process) 1
  V simple test data (in process) 1
  - Experiments
    - Crack and Fold Detector
  - Hierarchical training 2
    - Per layer training 2
    - Consider adding smoothness 2
    - Locking specific layers 2
    - Generic Looping 3
  - Data Augmentation (Need to be smart)
    - Flipping 2
    - Rotations 2
    - Translations 3
    - Elastic Deformations 3
- Visualization 1
  V Optical flow 1
  - Tensorboard 2
- Inference 3
  - Compute flow 3
  - Apply 3
  - Compute Confidence level
