<h2>TensorFlow-FlexUNet-Image-Segmentation-STARE-Retinal-Vessel (2025/07/01)</h2>

This is the first experiment of Image Segmentation for STARE Retinal Vessel 
 based on our TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and, <a href="https://drive.google.com/file/d/1dpyrsehyzNOgRgzv-PmhNAuHjLAoZY-U/view?usp=sharing">
Augmented-STARE-PNG-ImageMask-Dataset.zip</a>, 
which was derived by us from the following images and labels:<br><br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar">
<b>
Twenty images used for experiments
</b>
</a>
<br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar">
<b>
Hand labeled vessel network provided by Adam Hoover
</b>
</a>
<br>
<br>
On detail of <b>STARE(STructured Analysis of the Retina)</b>, 
please refer to the official site:<br>
<a href="https://cecas.clemson.edu/~ahoover/stare/">
STructured Analysis of the Retina
</a>
, and github repository <a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md">
STARE
</a>
<br><br>
As demonstrated in <a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-ETIS-LaribPolypDB">TensorFlow-FlexUNet-Image-Segmentation-ETIS-LaribPolypDB</a> ,
 our Multiclass TensorFlowFlexUNet, which uses categorized masks, can also be applied to 
single-class image segmentation models. 
This is because it inherently treats the background as one category and your single-class mask data as 
a second category. In essence, your single-class segmentation model will operate with two categorized classes within our Multiclass UNet framework.
<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the PNG dataset appear 
similar to the ground truth masks, but lack precision in some areas,
especially, failed to detect the thin retinal vessels.  
To improve segmentation accuracy, we could consider using a different segmentation model better suited for this task.
Please see also our experiment 
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel
</a><br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/10.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/barrdistorted_1003_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/barrdistorted_1003_0.3_0.3_8.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/barrdistorted_1003_0.3_0.3_8.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/distorted_0.03_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/distorted_0.03_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/distorted_0.03_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
</tr>
</table>

<hr>

<br>

<h3>1. Dataset Citation</h3>
The dataset used here has been take from 
from the following images and labels
in <a href="https://cecas.clemson.edu/~ahoover/stare/">
STructured Analysis of the Retina
</a>
:<br><br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/stare-images.tar">
<b>
Twenty images used for experiments
</b>
</a>
<br>
<a href="https://cecas.clemson.edu/~ahoover/stare/probing/labels-ah.tar">
<b>
Hand labeled vessel network provided by Adam Hoover
</b>
</a>
<br>
<br>
Please see also <a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md">
STARE
</a>
<br>
<br>
<b>Authors and Institutions</b><br>
Adam Hoover (Department of Electrical and Computer Engineering, Clemson University)<br>
Valentina Kouznetsova (Vision Computing Lab, Department of Electrical and Computer Engineering, <br>
University of California, San Diego, La Jolla)<br>
Michael Goldbaum (Department of Ophthalmology, University of California, San Diego)
<br>
<br>
<b>Citation</b><br>
@ARTICLE{845178,<br>
  author={Hoover, A.D. and Kouznetsova, V. and Goldbaum, M.},<br>
  journal={IEEE Transactions on Medical Imaging}, <br>
  title={Locating blood vessels in retinal images by piecewise threshold probing of a matched filter response}, <br>
  year={2000},<br>
  volume={19},<br>
  number={3},<br>
  pages={203-210},<br>
  doi={10.1109/42.845178}}<br>
<br><br>
<h3>
<a id="2">
2 STARE ImageMask Dataset
</a>
</h3>
 If you would like to train this STARE Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1dpyrsehyzNOgRgzv-PmhNAuHjLAoZY-U/view?usp=sharing">
Augmented-STARE-PNG-ImageMask-Dataset.zip</a>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─STARE
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
 
<b>STARE Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/STARE/STARE_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.

<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained STARETensorflowUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/STARE/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/STAREand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowFlexUNet.py">TensorflowFlesUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
num_classes    = 2

dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.You may train this model by setting this generator parameter to True. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b>Mask RGB_map</b><br>
[mask]
<pre>
mask_datatype    = "categorized"
mask_file_format = ".png"
;STARErgb color map dict for 1+1 classes.
;Background:black, Vessel:white
rgb_map = {(0,0,0):0,(255, 255, 255):1, }
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at middlepoint (epoch 44,45,46)</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 89,90,91)</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 91 by EarlyStopping callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/train_console_output_at_epoch_91.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/STARE/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/STARE/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/STARE/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/STARE/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/STARE</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for STARE.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/evaluate_console_output_at_epoch_91.png" width="720" height="auto">
<br><br>Image-Segmentation-STARE

<a href="./projects/TensorFlowFlexUNet/STARE/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this STARE/test was very low, and dice_coef very high as shown below.
<br>
<pre>
ategorical_crossentropy,0.0683
dice_coef_multiclass,0.9677
</pre>
<br>

<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/STARE</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for STARE.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/STARE/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks </b><br>

<table>
<tr>
<th>Image</th>b
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/10.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/10.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/barrdistorted_1001_0.3_0.3_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/barrdistorted_1001_0.3_0.3_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/barrdistorted_1001_0.3_0.3_15.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/deformed_alpha_1300_sigmoid_8_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/deformed_alpha_1300_sigmoid_8_2.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/deformed_alpha_1300_sigmoid_8_2.png" width="320" height="auto"></td>
</tr>


<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/distorted_0.03_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/distorted_0.03_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/distorted_0.03_rsigma0.5_sigma40_5.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/distorted_0.03_rsigma0.5_sigma40_18.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/distorted_0.03_rsigma0.5_sigma40_18.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/distorted_0.03_rsigma0.5_sigma40_18.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/images/hflipped_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test/masks/hflipped_15.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/STARE/mini_test_output/hflipped_15.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. Locating Blood Vessels in Retinal Images</b><br>
by Piecewise Threshold Probing of a<br>
Matched Filter Response<br>
Adam Hoover, Valentina Kouznetsova, and Michael Goldbaum<br>

<a href="https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf">
https://www.uhu.es/retinopathy/General/000301IEEETransMedImag.pdf
</a>
<br>
<br>
<b>2. STructured Analysis of the Retina</b><br>
<a href="https://cecas.clemson.edu/~ahoover/stare/">https://cecas.clemson.edu/~ahoover/stare/
</a>
<br>
<br>
<b>3. STARE</b><br>
<a href="https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md">
https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/STARE.md
</a>
<br>
<br>
<b>4. State-of-the-art retinal vessel segmentation with minimalistic models</b><br>
Adrian Galdran, André Anjos, José Dolz, Hadi Chakor, Hervé Lombaert & Ismail Ben Ayed<br>
<a href="https://www.nature.com/articles/s41598-022-09675-y">
https://www.nature.com/articles/s41598-022-09675-y
</a>
<br>
<br>
<b>5. Retinal blood vessel segmentation using a deep learning method based on modified U-NET model</b><br>
Sanjeewani, Arun Kumar Yadav, Mohd Akbar, Mohit Kumar, Divakar Yadav<br>
<a href="https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3">
https://www.semanticscholar.org/reader/f5cb3b1c69a2a7e97d1935be9d706017af8cc1a3</a>
<br>
<br>

<b>6, Tensorflow-Image-Segmentation-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-Retinal-Vessel</a>
<br>
<br>
<b>7. Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Pre-Augmented-STARE-Retinal-Vessel
</a>
<br>


