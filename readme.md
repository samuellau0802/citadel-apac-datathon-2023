## Report Outline
Our ultimate target is to investigate how to allocate traffic policy resources to reduce the effect of car crashes within the city.
Q1: 
Explore the occurrence of car crashes visually from geographic locations and also conduct some hypothesis tests to examine some potential influential features
Q2:
 Investigate deeper into these specific features, which will involve more severe car crashes through regression models
Q3:
 Combining previous findings to construct a mathematical model to determine the allocation of the traffic police station

### EDA, Visualization
1. Heatmap
2. Feature Importance 
3. Map visuals

### Hypothesis Testing
- We selectively choose 1-2 features and test their correlation to car crashes (severity/frequency)
(hypothesis tests (e.g., t-test) for ROAD_CONDITION, ILLUMINATION with respect to the number of car crashes (y label))

### Data Preprocessing
- Link car crash, crime, and traffic stop data using police district
- each input data represent the stats of a police district in a period of time (by month/weekday?)

### Model Fitting
1. x: feature matrix from the data preprocessing part
2. y: number of car crashes/car crash severity level

### Optimization
- Based on the number of car crashes and the severity level, how could we effectively allocate the traffic police in each police district?
