# **Project  Insights**

***

## About  the  Project
Omitted



## Current Achievement

* Collect Data of same_module_same_mac, same_module_diff_mac, diff_module_diff_mac(mini5&43)
* Generate pre-split Train/Test set  for each dataset
* A data processing program including converting csv/txt data to h5 data

* Experiment on each mini5 dataset




##  Discussion

***

## Switch to h5 file

The old code load data from csv/txt, which cost a lot of time to read and convert string data to float32. Now we load data from h5 which data is in the right format and stored continuously, it's proved that this method is far more quicker than csv/txt method.

**H5 On My Laptop**

```
Compute Cost 33.6818323135376s
Read Batch Cost 2.616137742996216s
Compute Cost 12.403181314468384s
Read Batch Cost 2.4905309677124023s
Compute Cost 12.3427894115448s
Read Batch Cost 2.472379684448242s
Compute Cost 12.368764162063599s
Read Batch Cost 2.4715495109558105s
Compute Cost 12.148866415023804s
Read Batch Cost 2.48537015914917s
```

Only 15% time was cost on the reading data.

In another computer, one iteration cost about 55s~60s, which indicates that in csv/txt method, 75% of time was consumed on File I/O.

However, still we hope to find some quicker way to read data, for disk of google platform is not as good as my computer. Comparing to GPU compute time, I/O still tack too much time.

**On GCP**

```
Compute Cost 10.438929080963135s
Read Batch Cost 5.137465715408325s
Compute Cost 0.8668842315673828s
Read Batch Cost 4.9810919761657715s
Compute Cost 0.8627252578735352s
Read Batch Cost 4.955933332443237s
Compute Cost 0.859318733215332s
Read Batch Cost 4.859827518463135s
Compute Cost 0.8686928749084473s
Read Batch Cost 4.979849338531494s
```



***

## Data Processing Speed

![Disk R&R VS Compute](D:\Win\Doc\#Project\RFML\doc\Disk R&R VS Compute.png)

Most of processing time was cost on reading csv and writing txt.

By reduce the string length of txt, there is some enhancement at txt writing.

Still, the disk I/O is the limitation of out program, powerful CPU or multiprocessing give no help. Try splitting the data set to several computer.



***

## Test Index

Micro Average Precision(Accuracy)

Macro Average Precision

Macro Average F1 Score (Harmonic average of Recall and Accuracy) 

Using Macro-F1 Score as final test index

Under normal circumstance, difference of this three indexes are minor.



***
## Result of  Diff Module Same Mac on Mini5

![](D:\Win\Doc\#Project\RFML\doc\diff_module_same_mac_mini5_final_test_result.png)

Classes' Macro F1 Score
0.9807583625073948

---

Classes' Macro Precision
0.9807583625073948

***

Classes' Micro Precision
0.9807321772639692



The Confusion Matrix gives the which two class the model most easy to mix up. And the peek num gives the frequency shit of each module.

We can find the most easily mixed up module of each module is also the module with most close frequency shift. Now we can imply that the "Analog feature" is most composed of the frequency shift, or peek num.

In some class, like B, G, we can find that the difference between frequency shit is inversely proportional to  probability of model mixing up two classes.

How ever, in 43 module date set, the distribution range won't change, so each module will have more modules with very similar peek num. The accuracy of large dataset may be decline a lot.



***

## Result of  Same Module Diff Mac on Mini5

Still Bad. We use 2048 hidden state, the train accuracy went up to 97%+

![same_module_diff_mac_mini5_train_accuracy](D:\Win\Doc\#Project\RFML\doc\same_module_diff_mac_mini5_train_accuracy.png)

![same_module_diff_mac_mini5_train_loss](D:\Win\Doc\#Project\RFML\doc\same_module_diff_mac_mini5_train_loss.png)

However, it's overfitting:

![same_module_diff_mac_mini5_test_accuracy](D:\Win\Doc\#Project\RFML\doc\same_module_diff_mac_mini5_test_accuracy.png)

![same_module_diff_mac_mini5_test_loss](D:\Win\Doc\#Project\RFML\doc\same_module_diff_mac_mini5_test_loss.png)

Classes' Recall
[0.59552296 0.58317252 0.61945195 0.68853724 0.57661135]

**************************************
Classes' Precision
[0.61744698 0.55551471 0.58963997 0.7023622  0.60388036]
**************************************
Classes' F1 Score
[0.60628684 0.56900772 0.60417843 0.69538102 0.5899309 ]
**************************************
Classes' Macro F1 Score
0.6137688431414794

**************************************
Classes' Macro Precision
0.6137688431414794
**************************************
Classes' Micro Precision
0.6126592049401776

![confusion_matrix](D:\Win\Doc\#Project\RFML\python\log\tf.h5data.same_module_diff_mac.LSTM.log\final_test_result\confusion_matrix.png)