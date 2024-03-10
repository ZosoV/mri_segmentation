# MRI Segmentation: A Classical Approach

### TODOs

- [ ] Review ways to test the algorithm
- [ ] Try to improve a little bit the algorithm
- [ ] Make a table with the changes sometimes.
    - Some improvements must be subject to specific things
    - Try to keep an understanding of the problem with data (not only visually)
    - Perform a diagram of the proposed method and it's close best variant
        - The unique graph I'm gonna include and made hahaha OJO hpt
- [ ] Start to write and propose the 3D segmentation
- [ ] Check how the methods work hahahaha =P

- CONDIER: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio if needed


Idea for the report

- The 2D algorithm is gonna focus on accuracy over time
- The 3D algorithm is gonna prioritize time against accuracy as it's not gonna apply expensive steps as outsu and two-step labeling (it's better one step)
- Let's check ACU and RUC metrics to measure the performance.



Present results on time, dice or jaccard, confussion matrix, AUC and RUC if possible. Check it later
- Drive decision based on Jaccard Index, and keep other as alternatives
- [ ] leer como mismo se lidia con unbalance data precision or recall is important?
- [ ] leer que conchas era AUC and ROC estan relacionados

- Add a short mention of data exploration with the normalization step and the histogram view of overlapping frequencies

For the second questions, four algorithms
1. Only otsu + two pass algorithm + k-means 92 
2. Adding convex hull (to focus on the accuracy of the internal region) 93
3. Deal with the overlapping section between label 1, 2 and 3 94
4. Check the posibility of improving the coverage of the internal region with edges
instead of convex hull (inversing the found edge)

For the thrid algorithm,
1. Among the tries the best option is until 3, but considering execution and resources
    - Avoid to use otsu by setting a fixed threshold at hand (maybe)
    - The two-pass labelling algorithm is expensive (instead try one-pass algorithm)
    - Explain why that works in 3D too.
2. Try fussion channel ideas (or put as future works)
    - In order to leverage the information from different channels
    - Simple function technique get the edge on each channel
    - sum all the channels together each channel with a weight 
        - the weight is assigned with a normal distribution in the current center in the current position
        - if your are getting the weight of the channel 5 (the other parts) around will be weighted with less values progressively (given more priority to the current one)
        - I think we can do this with a matrix multiplication that is not too expensive


