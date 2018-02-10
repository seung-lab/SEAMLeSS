# Neuroflow
Compute Flow across sections and align.

# Run

Run the environment
```
nvidia-docker run -it --net=host \
      -v $(pwd):/Neuroflow
      -v /usr/people/$USER/.neuroglancer/secrets/:/root/.cloudvolume/secrets/ \
      -e GOOGLE_APPLICATION_CREDENTIALS='/root/.cloudvolume/secrets/google-secret.json' \
      cavelab:latest-gpu bash
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
  - data (in process) 1
  V simple test data (in process) 1
  - hierarchical training 2
- Visualization 1
  - Optical flow 1
  - Tensorboard images 1
- Experiments 2
  - Log data 2
  - Tensorboard 2
- Inference 3
  - Compute flow 3
  - Apply 3
