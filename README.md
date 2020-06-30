# Subgraph counts as moments 

It is conveninent to treat subgraph counts as moments; as in [this work](https://arxiv.org/abs/1701.00505) or [that work](https://arxiv.org/abs/2006.15738). 
However, while handling subgraphs that way makes a lot of formal analysis easier, it requires the computation on non-trivial factors and terms in applications. This Pyhton code helps to produce these factors and terms. The key functions here is, from two base graphs (F1 and F2): list all unlabelled graphs that can be built using the base graphs as building blocks (the set H_{F1,F2} in the papers referenced above); and for each such unlabeled graph, count in how many ways the base graph can be combined to make the unlabelled graph (the term c_H in the papers referenced above).
