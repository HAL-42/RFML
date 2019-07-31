# **Project  Insights**

***

## About  the  Project
Omitted



## Current Achievement

* Collect Data of same_module_same_mac, same_module_diff_mac, diff_module_diff_mac(5 module/mac)
* Generate pre-split Train/Test set  for each dataset
* A data processing program including converting csv/txt data to h5 data




##  Discussion

***

## Switch to h5 file

The old code load data from csv/txt, which cost a lot of time to read and convert string data to float32. Now we load data from h5 which data is in the right format and stored continuously, it's proved that this method is far more quicker than csv/txt method.

H5

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

***

