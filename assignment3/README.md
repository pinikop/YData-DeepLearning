ConvNet_Word_Embedding.ipynb was originally submitted, even though we knew the network was not learning.  
Only after submission, we realized that (part of) the problem was probably that we did not take into account the imbalanced dataset. Once we fixed the loss function such that it would weight the classes acording to the prior distribution, the network started learning.  
