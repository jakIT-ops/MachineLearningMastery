# Step 1. Define you problem 

## Problem Defintion Framework 

Step 1: What is the problem?
Step 2: Why does the problem need to be solved?
Step 3: How would I solve the problem?

# Step 2. Prepare your data.

## Data Preparation Process 

Step 1: Select Data
Step 2: Preprocess Data
Step 3: Transform Data

## Improve Model Accuracy with Data Pre-Processing. 

### Data Preparation 

Асуудлаа загварчлахын өмнө та түүхий мэдээллээ урьдчилан боловсруулах ёстой. Тусгай бэлтгэл нь таны байгаа өгөгдөл болон ашиглахыг хүсч буй машин сургалтын алгоритмуудаас шалтгаална. Заримдаа өгөгдлийг урьдчилан боловсруулах нь загварын нарийвчлалыг санаанд оромгүй сайжруулахад хүргэдэг. Энэ нь өгөгдөл дэх харилцааг хялбаршуулсан эсвэл тодорхойгүй болгосонтой холбоотой байж болох юм. Өгөгдөл бэлтгэх нь чухал алхам бөгөөд та загварын нарийвчлалыг нэмэгдүүлэх боломжтой эсэхийг мэдэхийн тулд өөрийн өгөгдөлд тохирсон өгөгдлийг урьдчилан боловсруулах алхмуудыг туршиж үзэх хэрэгтэй.

* Add attributes to your data

> Нарийвчилсан загварууд нь нарийн төвөгтэй шинж чанаруудаас харилцааг гаргаж авах боломжтой боловч зарим загвар нь эдгээр харилцааг тодорхой бичихийг шаарддаг. Загварын үйл явцад оруулах сургалтын өгөгдлөөсөө шинэ шинж чанаруудыг гарган авах нь танд загварын гүйцэтгэлийг нэмэгдүүлэх болно.


* Remove attributes from your data

Some methods perform poorly with redundant or duplicate attributes. You can get a boost in model accuracy by removing attributes from your data.

* Transform attributes in your data

Transformations of training data can reduce the skewness of data as well as the prominence of outliers in the data. Many models expect data to be transformed before you can apply the algorithm.

## Discover Feature Engineering, How to Engineer Features and how to Get Good at it 

> Онцлогын инженерчлэл нь түүхий өгөгдлийг урьдчилан таамаглах загваруудын үндсэн асуудлыг илүү сайн илэрхийлэх онцлог болгон хувиргах үйл явц бөгөөд үүний үр дүнд үл үзэгдэх өгөгдлийн загварын нарийвчлалыг сайжруулдаг.

You can see the dependencies in this definition:

* The performance measures you’ve chosen (RMSE? AUC?)
* The framing of the problem (classification? regression?)
* The predictive models you’re using (SVM?)
* The raw data you have selected and prepared (samples? formatting? cleaning?

## 8 Tactics to Combat Imbalanced CLasses in your Machine learning dataset.

> Тэнцвэргүй өгөгдөл нь ангиудыг тэгш төлөөлдөггүй ангиллын асуудалтай холбоотой асуудлыг хэлдэг. Жишээлбэл, та 100 тохиолдол (мөр) бүхий 2 ангиллын (хоёртын) ангиллын асуудалтай байж болно. Нийт 80 тохиолдлыг Ангилал-1, үлдсэн 20 тохиолдлыг Ангилал-2 гэж тэмдэглэсэн байна. Энэ нь тэнцвэргүй өгөгдлийн багц бөгөөд Ангилал-1 болон Ангилал-2-ын харьцаа 80:20 буюу түүнээс дээш 4:1 байна.

We now understand what class imbalance is and why it provides misleading classification accuracy.

1. You can Collect More Data. 

2. Try Changing Your Performance Metric 

Accuracy is not the metric to use when working with an imbalanced dataset. We have seen that it is misleading.

There are metrics that have been designed to tell you a more truthful story when working with imbalanced classes.

3. Try Pesampling Your Dataset 

4. Try Generate Sunthetic Samples 

5. Try Different Algorithms.

6. Try Penalized Models.

7. Try a Different Perspective.

8. Try Getting Creative.

## Data Leakage in ML

Урьдчилан таамаглах загварчлалын зорилго нь сургалтын явцад үл үзэгдэх шинэ өгөгдөл дээр үнэн зөв таамаглал гаргах загварыг боловсруулах явдал юм.

### What is Data Leakage in ML

Өгөгдлийн алдагдал нь таныг хэт өөдрөг үзэлтэй, бүрэн хүчингүй урьдчилан таамаглах загваруудыг бий болгоход хүргэдэг.

Загвар үүсгэхийн тулд сургалтын өгөгдлийн багцаас гадуурх мэдээллийг ашиглах үед өгөгдөл алдагдсан байдаг. Энэхүү нэмэлт мэдээлэл нь загварт мэдэхгүй байсан зүйлийг сурах эсвэл мэдэх боломжийг олгож, улмаар барьж буй горимын тооцоолсон гүйцэтгэлийг хүчингүй болгож чадна.

## Techniques To Minimize DATA Leakage When Building Models 

1. Perform data preparation within your cross validation folds.
2. Hold back a validation dataset for final sanity check of your developed models.

# Step 3. Spot-check algorithms.

Spot-checking алгоритмууд нь таны машин сургалтын асуудалд олон янзын алгоритмуудыг хурдан үнэлж, ямар алгоритм дээр анхаарлаа төвлөрүүлж, юуг хаяхаа мэдэхэд оршино.

### Benefits of Spot-Checking Algorithms

Speed, Objective, Results.

### Top 10 Algorithms. 

Энэ нь таны дараагийн машин сургалтын асуудлыг шалгахын тулд алгоритмуудын товч жагсаалтыг эхлүүлэхэд тохиромжтой баримт бичиг байж магадгүй юм. Баримт бичигт жагсаасан өгөгдөл олборлох шилдэг 10 алгоритмууд байв.

* C4.5 This is a decision tree algorithm and includes descendent methods like the famous C5.0 and ID3 algorithms. 
* k-means. The go-to clustering algorithm.
* Support Vector Machines. This is really a huge field of study.
* Apriori. This is the go-to algorithm for rule extraction.
* EM. Along with k-means, go-to clustering algorithm.
* PageRank. I rarely touch graph-based problems.
* AdaBoost. This is really the family of boosting ensemble methods.
* knn (k-nearest neighbor). Simple and effective instance-based method.
* Naive Bayes. Simple and robust use of Bayes theorem on data.
* CART (classification and regression trees) another tree-based method.


## How To Choose The Right Test Options When Evaluating Machine Learning Algorithms

## A Data-Driven Approach to Choosing Machine Learning Algorithms 

# Step 4. Improve results.

## Machine Learning Performance Improvement Cheat Sheet

1. Improve Performance With Data.
2. Improve Performance With Algorithms.
3. Improve Performance With Algorithm Tuning.
4. Improve Performance With Ensembles.

# Step 5. Present results.


