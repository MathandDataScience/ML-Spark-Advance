# ML-Spark-Advance
## Machine Learning Models for Prediction of Spark Advance Tables of Spark-Ignited Engines

### Introduction
In spark-ignited (SI) combustion engine, spark advance (SA) is used to increase the efficiency and lower emissions. An engine control unit (ECU) uses a look up table for SA values under different running condition of the engine.  In order to create the SA tables an engine is bench tested; Where the engine is kept in steady conditions for several engine cycles during which SA values are feed to the ECU and performance is monitored. This testing is time-consuming, and very expensive.[1] [2] This paper explores the uses of Machine learning models to predict the values of a SA table for any vehicle using a SI engine.

An there are two types major types of combustion engine, two cycle and four-cycle engines. The data uses in the experiment came from four cycle engines and there SA works from the following principals. The main purpose of an engine is to convert energy to power. This is done by combustion of chemical energy to make mechanical energy like turning the wheels of a car. The combustion of an engine is linear and uses a crankshaft to convert that motion into a circular motion. This linear motion consists of four cycle of combustion. Theses cycles are intake, compression, power, and exhaust and happen approximately in the following deg ranges of the crankshaft intake (-360 to -180), compression (-180 to 0), power (0 to 180) exhaust (180-360). As we can see a full cycle happens in two rotations of the crank shaft. The area of focus related to SA is the middle 360 degrees (-180 to 180 as described previously). Shown in Figure 1 are the compression stroke and power stroke, the pistons location relates to a curve of pressure that builds in the cylinder. To completely burn the fuel let into the cylinder during the intake cycle, the spark of the combustion must start before the piston reaches top dead center TDC or 0 deg. This is due to the stochastic nature of combustion and operation of an engine.[3] The timing or degree at which the spark happens is denoted by the S in figure one. The location of S or the SA value is hard to determine mathematically and the SA value changes because the time it take to burn the fuel does not change but the engine speed and time to TDC does. A paper written by Kai Zhao and Yuhu Wu discusses the statistical and stochastic interconnections of spark advance (SA), crank angle at 50% burn of fuel (CA50) and indicated fuel conversion efficiency (IFCE). To summarize their work, the statistical relationships between SA and mean CA50 is a linear function and the relationships between CA50 and mean IFCE is a nonlinear function.[3] 

<p align="center">
 <img src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture1.jpg">
</p>
<p align="center">
 <em>Figure 1</em>
</p>


A common method of calibrated SA tables is base of Design of Experiment (DOE) this approach looks at the sensitive areas of the table and focus on running sweeps for that area and then statically filling in the rest.[4] Almost all traditional calibration methods require a bench test that are time consuming and involve offline analysis.[4] Another method proposed in a paper called Transient Spark Advance Calibration include adding extra sensors during bench test a limited number of sweeps and a ML model to produce an effective SA table. The outcome was an optimal SA based on combustion pressure with only running 9 sweeps and completing calibration in 10 mins.[4] The method this paper proposes is to find a ML model to produce an effective SA table with zero bench testing needed other then validation testing. 

### Methods
To conduct the proposed model exploration, data was collected from 26 vehicles that utilized a community supported Do-it yourself ECU called MegaSquirtII(MSQ)[5]. The data retrieved from the MSQ files was a 12 by 12 spark advance table similar to the one found in figure 2, where inputs RPM and % load was given along with an output of degrees of advancement. This data was then reshaped from 12 x 12 x 3 to 144 x 3. Hand collected data related to the engine was added, these data points included the number of cylinders (CYL), the displacement of the combustions chambers (DIS), size of the cylinder (BORE), and the compression ratio (COMP). These single data points were collected from the Wikipedia page relative to the engine code of the vehicles [6]. The single data point was then tiled to 144 data points making the shape of the dataset per vehicle to 144x7 and once all 26 vehicles were added together the data was shaped to 3744 x 7. 

<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture2.png">
</p>
<p align="center">
 <em>Figure 2:SA Table 1 from file 34.msq[5]</em>
</p>


To better understand the relationships of the data a correlation matrix plot and heatmap based around the degree of spark advance (Y) were generated and shown in Figure 3 and 4 respectively. Examining the correlation matrix plot, there is a correlation of Y and RPM that looks logarithmic, and the correlation Y and %  that looks to be decaying. The heatmap shows a deeper dimensional correlation between CYL and DIS, these relate to volume and in return could relate to the pressure in the cylinder. Due to this possibility a feature of DIS per CYL was engineered and added to the data set. See Figure 5 for updated Heatmap 

<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture3.png">
</p>
<p align="center">
 <em>Figure 3: correlation matrix plot </em>
</p>

<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture4.png">
</p>
<p align="center">
 <em>Figure 4: heatmap of data </em>
</p>

<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture5.png">
</p>
<p align="center">
 <em>Figure 5: New heatmap with added feature </em>
</p>

This dataset was then used to develop the models. A default model selector was set up with a linear model, linear model with regularization, a polynomial model to the 2nd  degree, a polynomial model with 10 degrees and regularization, a decision tree model and a random forest model, the model were trained and validated with a 10 Cross fold validation and a RMSE Bar graph was view for model selection. See figure 6. The nonlinear random forest model preformed the best and was selected for further development.  

<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture6.png">
</p>
<p align="center">
 <em>Figure 6: Model Section bar Graph of RMSE </em>
</p>

After further development of the random forest model trained and validated with a 10 Cross fold validation a RMSE value increased to 3.52%. Parameters and hyper-parameters can be found in the code. On prediction of the test set the random forest model obtained a RMSE of 4.52%. two heatmap were generated from the test data and the predicted test set, the data points that were used came from file “34.MSQ”, due to how the data was spilt and shuffled this only gave 33 point from file “34.MSQ”. One heatmap was created from of the “Y” values and one of the predicted “Y” values shown in Figure 7 and 8 respectively. 


<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture7.png">
</p>
<p align="center">
 <em>Figure 7: Heatmap Y test data, file 34.msq data points found in test set shown </em>
</p>

<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture8.png">
</p>
<p align="center">
 <em>Figure 8: Heatmap predicted Y test data, file 34.msq data points found in test set shown </em>
</p>

Development of a Shallow Neural Network (SNN)model, trained and validated with a 3 Cross fold validation got a RMSE value of 5.00%. This model used Ridge (L2) Regularization and two hidden layers. One layer used TANH at 120 nodes and other used ELU at 15 nodes. Other parameter can be found in the code. On prediction of the test set the SNN model obtained a RMSE of 5.13%. A heatmap was generated from the predicted Y values. The data points that were used to create the heat map came from file “34.MSQ” because the data was reshaped from all the file and made into one large data set then split with a shuffle the test set did not contain all the values of one given file.  Despite this a partial heat map can be generated.  see figure 9. 

<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture9.png">
</p>
<p align="center">
 <em>Figure 9: Heat map of predicted Y values from SNN model </em>
</p>

For final model developed was a Deep Neural Network (DNN). This model trained using batch normalization, L2 Regularization and Dropout, see code for more parameters and hyper-parameters. The DNN model obtained a validation RMSE of 7.04% and a test set RMSE of 5.27%. A prediction heatmap was generated as describe before and is shown in figure 10. 


<p align="center">
 <img width="50%" height="50%" src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture10-1.png">
</p>
<p align="center">
 <em>Figure 10: Heat map of predicted Y values from DNN model </em>
</p>

### Results 
The three models presented all obtain a low RMSE values, going only by this metric result of the random forest model on the test set of data was the highest at 4.53% for RMSE. A R-Squared value was calculated showing that the Random Forest model data has a 94% correction to the original data see figure 11. Further analyzes of the data and the R-Squared plot (figure 11) for random forest shows a good correlation and a decent correction for spread of data that is a bit weighted to values 25 and lower. This bias for higher value prediction in our application is actual more desired dues to the effiency of the engine at higher RPM is more critical than the lower end. Further analyze of the SNN model and it’s R-Squared plot(see figure 12) show that this model handles the bias better, this show by predicting the lower values are better than random forest but this comes at not prediction the rest of the data as well.  Resulting in a lower R-squared of 93%. 


<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture11.png">
</p>
<p align="center">
 <em>Figure 11:  R-Squared of Random forest model </em>
</p>


<p align="center">
 <img width="25%" height="25%" src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture12.png">
</p>
<p align="center">
 <em>Figure 12: R-Squared of SNN model </em>
</p>

The final model, DNN preformed the worst out of the three with a 92% R-squared value, see figure 13 for plot. Despite this I believe that with a larger data set and a further turning the DNN could perform better on the nonlinear components the combustion cycles.  As mention earlier to truly validate any of these models a single bench test would have to be performed for each model using the original data as a base line. 


<p align="center">
 <img  src="https://github.com/MathandDataScience/ML-Spark-Advance/blob/main/Pictures/Picture13.png">
</p>
<p align="center">
 <em>Figure 13: R-Squared of DNN model </em>
</p>


### Conclusion

Due to the nonlinear component the Random forest and SNN models performed the best, an may be an effective SA tables. As mention earlier to truly validate any of these models a single bench test would have to be performed for each model using the original data as a base line. The DNN model may perform better if there was a larger data set. Out of all these models the Random Forest shows the most promise given the current data set obtain a RMSE of 4.35% and a R-Squared of 94%. 

### References 
[1] K. Zhao, Y. Wu and T. Shen, "Beta-Distribution-Based Knock Probability Estimation, Control Scheme, and Experimental Validation for SI Engines," in IEEE Transactions on Control Systems Technology, vol.(early access),pp. 1-8, April 2020

[2]Enrico Corti, Nicolò Cavina, Alberto Cerofolini, Claudio Forte, Giorgio Mancini, Davide Moro, Fabrizio Ponti, Vittorio Ravaglioli, “Transient Spark Advance Calibration Approach,” Energy Procedia, Vol. 45, pp. 967-976, 2014.

[3] Zhang, Yahui, Gao, Jinwu, and Shen, Tielong. “Probabilistic Guaranteed Gradient Learning-Based Spark Advance Self-Optimizing Control for Spark-Ignited Engines.” IEEE Transactions on Neural Networks and Learning Systems, vol.29, no. 10 pp. 4683–4693, 2018.

[4] Enrico Corti, Nicolò Cavina, Alberto Cerofolini, Claudio Forte, Giorgio Mancini, Davide Moro, Fabrizio Ponti, Vittorio Ravaglioli, “Transient Spark Advance Calibration Approach,” Energy Procedia, Vol. 45, pp. 967-976, 2014.

[5] https://msqur.com/

[6] https://en.wikipedia.org/wiki/Main_Page



