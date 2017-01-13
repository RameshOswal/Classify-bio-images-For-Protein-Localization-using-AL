This folder contains the data for the Image Classification project for ABR 02-450/750 at CMU

There are 9 csv files, organized into three categories

"Easy"
EASY_TRAIN.csv: This is a 4120 by 27 matrix containing the features and labels for 4120 training images. 
		Each image is described by 26 features, which corresponds to the first 26 columns in the 
		matrix. The final column is the class label. Use this file as your pool (or stream) for active
		learning. 

EASY_TEST.csv: 	This is a 1000 by 27 matrix containing the features and labels for 1000 test images. 
		Each image is described by 26 features, which corresponds to the first 26 columns in the 
		matrix. The final column is the class label. Use this file to compute test errors.

EASY_BLINDED.csv:  This is a 250 by 27 matrix containing the features and labels for 250 test images. 
		   The first column is a unique id. The remaining 26 columns are the features. There is 
		   no label in this file. Use this file to make blinded predictions. Your predictions should
		   be in a text file named EASY_BLINDED.csv. Each line of that file should have the following
		   format:  <ID>, prediction


"Moderate"
MODERATE_TRAIN.csv: This is a 4120 by 27 matrix containing the features and labels for 4120 training images. 
		Each image is described by 26 features, which corresponds to the first 26 columns in the 
		matrix. The final column is the class label. Use this file as your pool (or stream) for active
		learning. 

MODERATE_TEST.csv: This is a 1000 by 27 matrix containing the features and labels for 1000 test images. 
		   Each image is described by 26 features, which corresponds to the first 26 columns in the 
		   matrix. The final column is the class label. Use this file to compute test errors.

MODERATE_BLINDED.csv:   This is a 250 by 27 matrix containing the features and labels for 250 test images. 
		   	The first column is a unique id. The remaining 26 columns are the features. There is 
		   	no label in this file. Use this file to make blinded predictions. Your predictions should
		   	be in a text file named MODERATE_BLINDED.csv. Each line of that file should have the following
		   	format:  <ID>, prediction



"Difficult"
DIFFICULT_TRAIN.csv: This is a 4120 by 53 matrix containing the features and labels for 4120 training images. 
		Each image is described by 26 features, which corresponds to the first 26 columns in the 
		matrix. The final column is the class label. Use this file as your pool (or stream) for active
		learning. 
DIFFICULT_TEST.csv: This is a 1000 by 53 matrix containing the features and labels for 1000 test images. 
		   Each image is described by 26 features, which corresponds to the first 26 columns in the 
		   matrix. The final column is the class label. Use this file to compute test errors.

DIFFICULT_BLINDED.csv:   This is a 250 by 53 matrix containing the features and labels for 250 test images. 
		   	The first column is a unique id. The remaining 26 columns are the features. There is 
		   	no label in this file. Use this file to make blinded predictions. Your predictions should
		   	be in a text file named DIFFICULT_BLINDED.csv. Each line of that file should have the following
		   	format:  <ID>, prediction
