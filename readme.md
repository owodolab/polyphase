# Phase Modelling using Hessians and Convex Envelopes

Code is writtern for Python 3.7

Dependencies are following:
```
autograd : pip install autograd
matplotlib (latest): python -m pip install -U matplotlib
plotly (latest) : pip install plotly==4.6.0
plotly static image export : conda install -c plotly plotly-orca==1.2.1 psutil request
numpy, scipy
```

To run the code that generates phase plots for different materials, in a shell command line:
```
cd /expts
python run_all_materials.py
```
Corresponding phase diagram is stored in `./data/paper_materials.py`
