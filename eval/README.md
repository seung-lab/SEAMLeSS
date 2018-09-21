# Evaluation  
Toolkit to assess alignment quality.

## Tools   
**cpc** Create an image by scoring overlapping chunks in neighboring 
sections with Pearson R (chunked pearson correlation).

**inspect** Inspect the vector fields produced by SEAMLeSS.

## How to use  
### cpc
For more information, use argparse help, e.g.  
```
python cpc.py -h
```

### inspect
Inspect is based on the python state server interface to Neuroglancer.

1. `cd eval/inspect`
1. Open python

       python

1. From within the REPL, import the Field class and direct it to the parent 
directory to the image & vec CloudVolumes In the example below, we point to the 
field output for a run on the pinky100 dataset.

       from field import Field
       path = "gs://neuroglancer/seamless/matriarch_tile7_pinky100_pairs_write_res_v13"
       F = Field(path, mip=5)
       # open link provided, navigate to z=1345, & zoom in to an area of interest
       F.inspect(F.get_bbox())



