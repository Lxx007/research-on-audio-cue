# Source code for Toward DDA with audio cues by GPR in a FPS

Files here are data and source code of the following paper. For relevant gameplay video, see https://tinyurl.com/FPSaudioCues


Xiaoxu Li, Marcel Wira and Ruck Thawonmas, "Toward Dynamic Difficulty Adjustment with Audio Cues by Gaussian Process Regression in a First-Person Shooter," the 21st International Federation for Information Processing - International Conference on Entertainment Computing (IFIP-ICEC 2022), Bremen, Germany, Nov. 1-3, 2022.

DataBase.py is our data after collecting from our FPS game, which you can view the videos from the link shown above.
Based on the informed consent, data shared here have already completed the data desensitization.
Currently, we only open-sourced our data with game completion time which is used in our paper.

Based on the data provided here, we can simply calculate the mean value of time at each level.
At the same time, since the player is playing our game for the first time, we think that the player needs to be familiar with our game when he plays the first level. Therefore, in our paper, we focus on the performance of all players after removing the first level.

![Figure 1](https://user-images.githubusercontent.com/30626090/183309034-8b153983-7731-49ff-9e46-456abec1ce52.png)

In the DataBase.py file, PlayerID means every player's ID, PlayerTime is every player's completion time with 5 levels, player_seq is the order in which the levels are played by each player, and Level is our level's audio cue volume setting corresponding to our Level Num which is the index of this list.
Based on simple statistics, we can draw the above conclusions.

CodeOnExperiment.py explain our GPR process mentioned in the paper, and KernelFunction.py is our kernel library.
After executing this python file, we will get 8 output(savedfiles)
These files should be named as:

Trend_0_RBF_7.npy

Trend_1_RBF_7.npy

Trend_2_RBF_7.npy

Trend_3_RBF_7.npy

Slope_0_RBF_7.npy

Slope_1_RBF_7.npy

Slope_2_RBF_7.npy

Slope_3_RBF_7.npy

The SmapleOutput folder contains one set of sample outputs.
Since we use 4-fold cross validation, 0 1 2 3 means each fold Num.
File name Trend_X_Y_Z.npy means that this is the prediction of GPR with the kernel Y hyperparameter length scale Z for (X+1)'s fold.
In each Trend_X_Y_Z.npy file, we have 10 pieces of data.
For each piece of data, it is a data sequence with 51 predictions by GPR. And when the result R is smaller than the threshold = 1, for the next prediction we will skip and show the R. You can find more details in our paper.
For Slope_X_Y_Z.npy, this is a by-product of our experiment, we will not explain it here.

After collecting Trend_X_Y_Z.npy, it is not hard for us to calculate the performance of our algorithm.
With simple statistics, we can draw the follwing conclusions.

![400](https://user-images.githubusercontent.com/30626090/183308103-55bcd49b-8ba0-4a13-9d6b-a9aa713e6488.png)

Because our algorithm has some random properties, it is difficult to precisely restore the above figure, but after each code run, the trend of the data should be similar to the above figure.

Regarding baselines, form line 40 to line 47 on CodeOnExperiment.py, it is not hard to create baselines with Multi Polynomial Function provided by numpy. With these, and parameter mentioned on the paper, it is easy to reproduce our baselines.

Finally, based on the process mentioned above. After execute our algorithms with 10 times and calculated the means value. We will get the following table

![image](https://user-images.githubusercontent.com/30626090/183308808-7b4f0b75-d60d-444c-baf1-eece59750452.png)
