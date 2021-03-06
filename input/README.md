# Description of Net-4 and the corresponding results

## Parameters
<img src="https://latex.codecogs.com/svg.image?N" title="N" /> = 10000  
<img src="https://latex.codecogs.com/svg.image?d_{avg}" title="d_{avg}" /> = 5  
<img src="https://latex.codecogs.com/svg.image?d_{max}" title="d_{max}" /> = 100  
<img src="https://latex.codecogs.com/svg.image?\mu" title="\mu" /> = 0.1  
<img src="https://latex.codecogs.com/svg.image?c_{min}" title="c_{min}" /> = 200  
<img src="https://latex.codecogs.com/svg.image?c_{max}" title="c_{max}" /> = 1000  
<img src="https://latex.codecogs.com/svg.image?\tau_1" title="\tau_1" /> = 2  
<img src="https://latex.codecogs.com/svg.image?\tau_2" title="\tau_2" /> = 1  
<img src="https://latex.codecogs.com/svg.image?\alpha_1" title="\alpha_1" /> = 0.9  
<img src="https://latex.codecogs.com/svg.image?\alpha_2" title="\alpha_2" /> = 0.1

## Statistics
\# Nodes = 10,000  
\# Positive edges = 54,215(81.9%)  
\# Negative edges = 11,969(18.1%)  
\# Classes = 18

## Results
Due to the limited time and computing resources, we only report the results of several methods.
We will provide more experimental results in the final version.

### Net-4(<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />=10)
**DDRE**  AUC=.992 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .000  F1=.896 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .006

**GSGNS**  AUC=.991 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .000  F1=.868 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .006

**SLF**   AUC=.558 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .004  F1=.070 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .002

### Net-4(<img src="https://latex.codecogs.com/svg.image?\alpha" title="\alpha" />=5)
**DDRE**  AUC=.989 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .001  F1=.875 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .006

**GSGNS**  AUC=.990 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .000  F1=.852 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .004

**SLF**   AUC=.544 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .003  F1=.069 <img src="https://latex.codecogs.com/svg.image?\pm" title="\pm" /> .003
