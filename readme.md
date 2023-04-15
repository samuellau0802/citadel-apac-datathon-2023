## Report Outline
Our ultimate target is to investigate how to allocate traffic policy resources to reduce the effect of car crashes within the city.

Q1: Do high-frequency car crashes have prominent geographical concentration characteristics, and what are the potential impact factors?

To do: Explore the occurrence of car crashes visually from geographic locations and also conduct some hypothesis tests to examine some potential influential features

Q2: Considering the frequency of car crashes and the accident's severity, what factors will affect it?

Q2(by lsy): Does the reponse time of traffic police have effect on the accidents' severity?
To do(by lsy): do some hypothesis testing. If it does, try to find the quantitative relation between these two.

To do: Investigate deeper into these specific features, which will involve more severe car crashes through regression models

Q3: How can we provide effective recommendations on how to improve road safety within Philadelphia?

To do: Combining previous findings to construct a mathematical model to determine the allocation of the traffic police station

### EDA, Visualization
1. Heatmap
2. Map visuals
3. Feature Importance 
### Hypothesis Testing
- We selectively choose 1-2 features and test their correlation to car crashes (severity/frequency)
(hypothesis tests (e.g., t-test) for like ROAD_CONDITION, ILLUMINATION, WEATHER with respect to the number of car crashes (y label))

### Data Preprocessing
- Link car crash, crime, and traffic stop data using police district
- each input data represent the stats of a police district in a period of time (by month/weekday)

### Model Fitting
1. x: feature matrix from the data preprocessing part
2. y: number of car crashes/car crash severity level

### Optimization
- Based on the number of car crashes and the severity level, how could we effectively allocate the traffic police in each police district?
