# ml_waves19

Hutch Research Project

## Steps Forward

### Steps with existing model framework (Hindcast ~ 10yrs, Forecast data ~ 18 months)

1. Test the relation of model skill and hindcast training length. E.g. 10-years vs 1-year, vs 3-month. What length is required for "good" skill.
2. Test generalizeability of model skill across all available buoys with sufficient data (N~15). This would look like a matrix with each model evaluated on each buoy. Jonny did a 3x3 version of this in his paper.  
3. Test skill of pre-training with nearby buoy (using guidance from 2), and refinement training for locations with insufficient data (e.g. new buoys).
4. Test skill of hindcasts trained models on forecast data. Forecast data is in a different format and this will require some adaptation to the code.
5. Depending on results from 1 and 2, Test skill of models trained on forecasts and/or trained on hindcasts with forecast training refinement

Questions to answer
1. How much training data is sufficient, does it vary with location or frequency?
2. How generalizeable are models across a region? Does this depend on frequency or buoy mooring depth? Is pre-training and refining useful?
3. How skillful is the model with real forecast data? Is pre-training on hindcast data neccessary or useful?

### New model frameworks

1. N-buoy hindcast spectral model
  * Start with 2 buoys initially, suggest Gray's harbor and 46005 (far offshore buoy).
  * Utilize N-buoys. It will be challening to handle gaps, can we include some kind of null values where gaps are present in training?

2. N-buoy hindcast spectral model with extra available spatial information (wind obs, sofar buoy obs, etc).
  * Create framework for sofar buoy bulk parameter input
  * Create framework for buoy wind observations

3. Overlapping forecast model: forecasts generated every 6-hours for 7-days
 
