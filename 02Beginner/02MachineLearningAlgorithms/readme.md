# Algorithms Grouped by Learning Style

## Let’s take a look at three different learning styles in machine learning algorithms:

### 1. Supervised Learning.

Input data is called training data and has a known label or result such as spam/not-spam or a stock price at a time.

Загварыг урьдчилан таамаглах шаардлагатай сургалтын процессоор бэлтгэдэг бөгөөд тэдгээр таамаглал буруу байвал засдаг. Загвар нь сургалтын өгөгдөл дээр хүссэн нарийвчлалын түвшинд хүрэх хүртэл сургалтын үйл явц үргэлжилнэ.

Example problems are classification and regression.

Example algorithms include: Logistic Regression and the Back Propagation Neural Network

### 2. Unsupervised learning.

Input data is not labeled and does not have a known result.

Оролтын өгөгдөлд байгаа бүтцийг хасах замаар загварыг бэлтгэдэг. Энэ нь ерөнхий дүрмийг гаргаж авах зорилготой байж магадгүй юм. Энэ нь илүүдлийг системтэйгээр багасгахын тулд математик процессоор дамжсан эсвэл ижил төстэй байдлаар өгөгдлийг цэгцлэх замаар байж болно.

Example problems are clustering, dimensionality reduction and association rule learning.

Example algorithms include: the Apriori algorithm and K-Means.

### 3. Semi-Supervised Learning

Input data is a mixture of labeled and unlabelled examples

Урьдчилан таамаглахыг хүссэн асуудал байгаа боловч загвар нь өгөгдлийг зохион байгуулах, түүнчлэн таамаглал гаргах бүтцийг сурах ёстой.

Example problems are classification and regression.

Example algorithms are extensions to other flexible methods that make assumptions about how to model the unlabeled data.

## Regression Algorithms.

Регресс нь загвараар хийсэн таамаглал дахь алдааны хэмжигдэхүүнийг ашиглан давталттайгаар сайжруулсан хувьсагчдын хоорондын хамаарлыг загварчлахад чиглэгддэг.

Regression methods are a workhorse of statistics and have been co-opted into statistical machine learning. This may be confusing because we can use regression to refer to the class of problem and the class of algorithm. Really, regression is a process.



The most popular regression algorithms are:

* Ordinary Least Squares Regression (OLSR)
* Linear Regression
* Logistic Regression
* Stepwise Regression
* Multivariate Adaptive Regression Splines (MARS)
* Locally Estimated Scatterplot Smoothing (LOESS)

## Instance-based Algorithms

Жагсаалд суурилсан сургалтын загвар нь тухайн загварт чухал эсвэл шаардлагатай гэж үзсэн сургалтын өгөгдлийн жишээ эсвэл жишээнүүдийн шийдвэрийн асуудал юм.

Ийм аргууд нь ихэвчлэн жишээ өгөгдлийн санг бүрдүүлж, хамгийн сайн тохирохыг олж таамаглахын тулд ижил төстэй байдлын хэмжүүр ашиглан шинэ өгөгдлийг мэдээллийн сантай харьцуулдаг. Ийм учраас жишээнд суурилсан аргуудыг ялагч-бүгдийг-бүгдийг авах арга, санах ойд суурилсан сургалт гэж нэрлэдэг. Хадгалсан тохиолдлуудын төлөөлөл болон жишээнүүдийн хооронд ашигласан ижил төстэй байдлын хэмжүүрт анхаарлаа хандуулдаг.

The most popular instance-based algorithms are:

* k-Nearest Neighbor (kNN)
* Learning Vector Quantization (LVQ)
* Self-Organizing Map (SOM)
* Locally Weighted Learning (LWL)
* Support Vector Machines (SVM)

## Regularization Algorithms

Загваруудыг нарийн төвөгтэй байдлаас нь хамааруулан шийтгэдэг өөр аргад (ихэвчлэн регрессийн аргууд) хийсэн өргөтгөл, ерөнхийлөлт хийхэд илүү хялбар энгийн загваруудыг илүүд үздэг. 

Эдгээр нь түгээмэл, хүчирхэг бөгөөд бусад аргуудад хийгдсэн ерөнхийдөө энгийн өөрчлөлтүүд учраас би зохицуулалтын алгоритмуудыг тусад нь жагсаасан болно.

* Ridge Regression
* Least Absolute Shrinkage and Selection Operator (LASSO)
* Elastic Net
* Least-Angle Regression (LARS)


## Decision tree algorithms

Decision tree methods construct a model of decisions made based on actual values of attributes in the data.

Өгөгдсөн бичлэгийн хувьд урьдчилан таамаглах шийдвэр гарах хүртэл модны бүтцэд шийдвэрүүд салаа. Шийдвэрийн модыг ангилал болон регрессийн асуудалд зориулсан өгөгдөлд сургадаг. Шийдвэрийн мод нь ихэвчлэн хурдан бөгөөд үнэн зөв байдаг бөгөөд машин сурахад хамгийн дуртай байдаг.

* Classification and Regression Tree (CART)
* Iterative Dichotomiser 3 (ID3)
* C4.5 and C5.0 (different versions of a powerful approach)
* Chi-squared Automatic Interaction Detection (CHAID)
* Decision Stump
* M5
* Conditional Decision Trees

## Bayesian Algorithms

Bayesian methods are those that explicitly apply Bayes’ Theorem for problems such as classification and regression.

* Naive Bayes
* Gaussian Naive Bayes
* Multinomial Naive Bayes
* Averaged One-Dependence Estimators (AODE)
* Bayesian Belief Network (BBN)
* Bayesian Network (BN)


## Clustering Algorithms

Clustering, like regression, describes the class of problem and the class of methods.

Кластерын аргуудыг ихэвчлэн төв дээр суурилсан, шаталсан зэрэг загварчлалын аргуудаар зохион байгуулдаг. Бүх аргууд нь өгөгдлийг хамгийн их нийтлэг байдлын бүлгүүдэд хамгийн сайн зохион байгуулахын тулд өгөгдлийн өвөрмөц бүтцийг ашиглахад чиглэгддэг.

* k-Means
* k-Medians
* Expectation Maximisation (EM)
* Hierarchical Clustering

## Association Rule Learning Algorithms

Association rule learning methods extract rules that best explain observed relationships between variables in data. 

Эдгээр дүрмүүд нь байгууллага ашиглаж болох олон хэмжээст өгөгдлийн багц дахь чухал, арилжааны ач холбогдолтой холбоог олж илрүүлж чадна.

* Apriori algorithm

* Eclat algorithm

## Artificial Neural Network Algorithms

Artificial Neural Networks are models that are inspired by the structure and/or function of biological neural networks

Эдгээр нь регресс болон ангиллын асуудалд түгээмэл хэрэглэгддэг загвар тааруулах анги боловч үнэхээр бүх төрлийн асуудлын төрөлд зориулсан олон зуун алгоритм, хувилбаруудаас бүрдсэн асар том дэд талбар юм. Энэ салбарт асар их өсөлт, алдартай болсон тул би Deep Learning-ийг мэдрэлийн сүлжээнээс салгасан гэдгийг анхаарна уу. Энд бид илүү сонгодог аргуудын талаар санаа зовж байна.

* Perceptron
* Multilayer Perceptrons (MLP)
* Back-Propagation
* Stochastic Gradient Descent
* Hopfield Network
* Radial Basis Function Network (RBFN)

## Deep Learning Algorithms.

Deep Learning methods are a modern update to Artificial Neural Networks that exploit abundant cheap computation.

Тэд илүү том, илүү төвөгтэй мэдрэлийн сүлжээг бий болгоход санаа тавьдаг бөгөөд дээр дурдсанчлан олон аргууд нь зураг, текст гэх мэт шошготой аналог өгөгдлийн маш том мэдээллийн багцтай холбоотой байдаг. аудио, видео.

* Convolutional Neural Network (CNN)
* Recurrent Neural Networks (RNNs)
* Long Short-Term Memory Networks (LSTMs)
* Stacked Auto-Encoders
* Deep Boltzmann Machine (DBM)
* Deep Belief Networks (DBN)

## Dimensionality Reduction Algorithms

Кластерын аргуудын нэгэн адил хэмжээст байдлыг багасгах нь өгөгдлийн төрөлхийн бүтцийг эрэлхийлж, ашигладаг боловч энэ тохиолдолд хяналтгүй байдлаар эсвэл бага мэдээлэл ашиглан өгөгдлийг нэгтгэн дүгнэх, дүрслэх зорилгоор ашигладаг. Энэ нь хэмжээст өгөгдлийг дүрслэн харуулах эсвэл хяналттай сургалтын аргад ашиглаж болох өгөгдлийг хялбарчлахад тустай байж болно. Эдгээр аргуудын ихэнхийг ангилал, регрессэд ашиглахад тохируулан өөрчилж болно.

* Principal Component Analysis (PCA)
* Principal Component Regression (PCR)
* Partial Least Squares Regression (PLSR)
* Sammon Mapping
* Multidimensional Scaling (MDS)
* Projection Pursuit
* Linear Discriminant Analysis (LDA)
* Mixture Discriminant Analysis (MDA)
* Quadratic Discriminant Analysis (QDA)
* Flexible Discriminant Analysis (FDA)

## Ensemble Algorithms

Ensemble methods are models composed of multiple weaker models that are independently trained and whose predictions are combined in some way to make the overall prediction.

Ямар төрлийн сул суралцагчдыг нэгтгэх, тэдгээрийг хэрхэн хослуулах талаар ихээхэн хүчин чармайлт гаргадаг. Энэ бол маш хүчирхэг арга техник бөгөөд иймээс маш их алдартай.

* Boosting
* Bootstrapped Aggregation (Bagging)
* AdaBoost
* Weighted Average (Blending)
* Stacked Generalization (Stacking)
* Gradient Boosting Machines (GBM)
* Gradient Boosted Regression Trees (GBRT)
* Random Forest

## Othe Machine Learning Algorithms 

Feature selection algorithms
Algorithm accuracy evaluation
Performance measures
Optimization algorithms

I also did not cover algorithms from specialty subfields of machine learning, such as:

Computational intelligence (evolutionary algorithms, etc.)
Computer Vision (CV)
Natural Language Processing (NLP)
Recommender Systems
Reinforcement Learning
Graphical Models
And more…

# Gentle Introduction to the Bias-Variance Trade-Off in Machine Learning

Supervised machine learning algorithms can best be understood through the lens of the bias-variance trade-off.

Аливаа хяналттай машин сургалтын алгоритмын зорилго нь оролтын өгөгдөл (X) өгөгдсөн гаралтын хувьсагчийн (Y) зураглалын функцийг (f) хамгийн сайн тооцоолох явдал юм. Mapping функцийг ихэвчлэн зорилтот функц гэж нэрлэдэг, учир нь энэ нь өгөгдсөн хяналттай машин сургалтын алгоритмын ойролцоологдохыг зорьдог функц юм.

* Bias Error
* Variance Error
* Irreducible Error

Аль алгоритмыг ашиглахаас үл хамааран бууруулж болохгүй алдааг багасгах боломжгүй. Энэ нь асуудлын сонгосон хүрээнээс гарсан алдаа бөгөөд оролтын хувьсагчийг гаралтын хувьсагч руу буулгахад нөлөөлдөг үл мэдэгдэх хувьсагч зэрэг хүчин зүйлсээс үүдэлтэй байж болно.

## Bias Error 

Bias are the simplifying assumptions made by a model to make the target function easier to learn.

Ерөнхийдөө шугаман алгоритмууд нь өндөр хазайлттай байдаг бөгөөд тэдгээрийг сурахад хурдан, ойлгоход хялбар боловч ерөнхийдөө уян хатан биш байдаг. Хариуд нь алгоритмын хазайлтыг хялбаршуулсан таамаглалд нийцэхгүй байгаа нарийн төвөгтэй асуудлууд дээр урьдчилан таамаглах чадвар багатай байдаг.

* `Low Bias:` Suggests less assumptions about the form of the target function.
* `High-Bias:` Suggests more assumptions about the form of the target function.

Examples of low-bias machine learning algorithms include: Decision Trees, k-Nearest Neighbors and Support Vector Machines.

Examples of high-bias machine learning algorithms include: Linear Regression, Linear Discriminant Analysis and Logistic Regression.

## Variance Error 

Variance is the amount that the estimate of the target function will change if different training data was used.

Зорилтот функцийг сургалтын өгөгдлөөс машин сургалтын алгоритмаар тооцдог тул алгоритм нь зарим нэг зөрүүтэй байх ёстой. Энэ нь нэг сургалтын өгөгдлийн багцаас нөгөөд хэт их өөрчлөлт хийх ёсгүй бөгөөд энэ нь алгоритм нь оролт ба гаралтын хувьсагчдын хоорондох далд далд зураглалыг сонгоход сайн гэсэн үг юм.

Өндөр зөрүүтэй машин сургалтын алгоритмууд нь сургалтын өгөгдлийн онцлогоос ихээхэн хамаардаг. Энэ нь сургалтын онцлог нь зураглалын функцийг тодорхойлоход ашигласан параметрийн тоо, төрөлд нөлөөлдөг гэсэн үг юм.

* `Low Variance:` Suggests small changes to the estimate of the target function with changes to the training dataset.
* `High Variance:` Suggests large changes to the estimate of the target function with changes to the training dataset.

Ерөнхийдөө уян хатан чанар сайтай шугаман бус машин сургалтын алгоритмууд нь маш их хэлбэлзэлтэй байдаг. Жишээлбэл, шийдвэрийн моднууд өндөр хэлбэлзэлтэй байдаг бөгөөд энэ нь модыг ашиглахаас өмнө тайрахгүй бол бүр ч өндөр байдаг.

Examples of low-variance machine learning algorithms include: Linear Regression, Linear Discriminant Analysis and Logistic Regression.

Examples of high-variance machine learning algorithms include: Decision Trees, k-Nearest Neighbors and Support Vector Machines.

## Bias-Variance Trade-Off

Хяналттай машин сургалтын аливаа алгоритмын зорилго нь бага хазайлт, бага хэлбэлзэлд хүрэх явдал юм. Хариуд нь алгоритм нь урьдчилан таамаглах сайн гүйцэтгэлд хүрэх ёстой.

* `Linear` machine learning algorithms often have a high bias but a low variance.
* `Nonlinear` machine learning algorithms often have a low bias but a high variance.

















