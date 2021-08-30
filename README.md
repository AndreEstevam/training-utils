# occupation-detection-on-satellite-imagery

The objective of this project is to adapt, train and test a Mask R-CNN model to identify occupations in satellite imagery for regulatory purposes. The state-of-the-art architecture in question was chosen because each occupation, even if reasonably clustered, must be independently identified and accurately delimitated, and region based approaches showed good results for detecting small targets.

The project dataset of training and test samples is limited, containing few km² of manually labelled occupations extracted via ArcGIS, so for that reason the model will be first trained on a publicly available dataset from <a href="https://www.aicrowd.com/challenges/mapping-challenge/">AIcrowd mapping challenge</a> and then finetuned on Brasília's dataset for better generalization.

The implementation was made through the framework Pytorch and later through a fork from leekunhee/Mask_RCNN repo, which is an adaptation from matterpot/Mask_RCNN to use tensorflow 2.0, for testing purposes.
