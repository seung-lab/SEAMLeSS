# Evaluation  
Toolkit to assess alignment quality.

## Tools   
**chunked** Create an image by scoring overlapping chunks in neighboring 
sections with a similarity metric. Currently, only pearson r is available. 

**inspect** Inspect the vector fields produced by Neuroflow.

## How to use  
### chunked
For more information, use argparse help, e.g.  
```
python chunked.py -h
```

### inspect
Inspect is based on the stateServer interface to Neuroglancer. This interface
is implemented in the Seung Lab fork of Neuroglancer which can be accessed at 
[https://neuromancer-seung-import.appspot.com](https://neuromancer-seung-import.appspot.com).

1. `pip install -r eval/inspect/requirements.txt`
1. `cd eval/inspect`
1. Open python

       python

1. From within the REPL, import the Field class and direct it to the 
Storage directory containing the CloudVolumes of both the x & y vector fields.
In the example below, we point to the Neuroflow output for a run on the FAFB
dataset.

       from field import Field
       F = Field("gs://neuroglancer/seamless/matriarch_tile7_drosophila_pairs_v0/vec", mip=5, port=9999)

1. You should see a statement `IOLoop starting`. You're now broadcasting a state server at this address: `https://localhost:9999`. The port kwarg of Field specifies the port in the address. 
1. This state server is not secure, so your browser requires permission to access it. Open your browser and go to your state server's address, `https://localhost:9999`, and grant permission to access that url.
1. Still in your browser, load up your dataset of interest in neuroglancer, including the URL to the state server you've created in its state. 
For example:
[https://neuromancer-seung-import.appspot.com/#!{'navigation':{'pose':{'position':{'voxelSize':[4_4_40]_'voxelCoordinates':[0_0_0]}_'orientation':[0_0_0.7071067690849304_0.7071067690849304]}_'zoomFactor':11.880292622753483}_'layout':'xy'_'stateServer':'https://localhost:9999'}](https://neuromancer-seung-import.appspot.com/#!{'navigation':{'pose':{'position':{'voxelSize':[4_4_40]_'voxelCoordinates':[0_0_0]}_'orientation':[0_0_0.7071067690849304_0.7071067690849304]}_'zoomFactor':11.880292622753483}_'layout':'xy'_'stateServer':'https://localhost:9999'})    
Note that the state server address is passed to the 'stateServer' key at the end of the URL.
1. Make a change to the state of your neuroglancer app, e.g. pan in the image.
1. Back in python, you should see a statement 'state initialized', along with a print out of the JSON describing the state.
1. You can now inspect the vector field.

       F.inspect(F.controller.get_bbox())

