Магадлал гэдэг нь тодорхой бус байдлыг тоолж, ашиглах математик юм. Энэ нь математикийн олон салбарын (статистик гэх мэт) үндэс суурь бөгөөд хэрэглээний машин сурахад чухал ач холбогдолтой юм

# 1. Discover what Probability is

## Basics of Mathematical Notation for Machine Learning

## 5 Reasons to Learn Probability for Machine Learning

1. Class Membership Requires Predicting a Probability

2. Models Are Designed Using Probability

3. Models Are Trained With Probalilistic Frameworks

4. Models Are Tuned With a Probabilistic Framework

5. Models Are Evaluated With Probabilistic Measures

# 2. Discover why Probability is so important for ml

# 3. Dive into Probability topics 

## Lesson 01: Probability and Machine Learnin

Probability and statistics help us to understand and quantify the expected value and variability of variables in our observations from the domain.
Probability helps to understand and quantify the expected distribution and density of observations in the domain.
Probability helps to understand and quantify the expected capability and variance in performance of our predictive models when applied to new data.

## Lesson 02: Three Types of Probability

Joint Probability.

* P(A and B) = P(A given B) * P(B)

Marginal Probability

* P(X=A) = sum P(X=A, Y=yi) for all y

Conditional Probability

* P(A given B)

* P(A given B) = P(A and B) / P(B)

## Lesson 03: Probability Distributions

`Discrete Random Variable.` Values are drawn from a finite set of states.

`Continuous Random Variable.` Values are drawn from a range of real-valued numerical values.


### Discrete Probability Distributions

Poisson distribution.
Bernoulli and binomial distributions.
Multinoulli and multinomial distributions.

### Continuous Probability Distributions.

Normal or Gaussian distribution.
Exponential distribution.
Pareto distribution.

### Randomly Sample Gaussian Distribution

```py
# sample a normal distribution
from numpy.random import normal
# define the distribution
mu = 50
sigma = 5
n = 10
# generate the sample
sample = normal(mu, sigma, n)
print(sample)
```

## Lesson 04: Naive Bayes Classifier

```py
# example of gaussian naive bayes
from sklearn.datasets import make_blobs
from sklearn.naive_bayes import GaussianNB
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
# define the model
model = GaussianNB()
# fit the model
model.fit(X, y)
# select a single sample
Xsample, ysample = [X[0]], y[0]
# make a probabilistic prediction
yhat_prob = model.predict_proba(Xsample)
print('Predicted Probabilities: ', yhat_prob)
# make a classification prediction
yhat_class = model.predict(Xsample)
print('Predicted Class: ', yhat_class)
print('Truth: y=%d' % ysample)
```

## Lesson 05: Entropy and Cross-Entropy

* Low Probability Event: High Information (surprising).
* High Probability Event: Low Information (unsurprising).


```py
# example of calculating cross entropy
from math import log2

# calculate cross entropy
def cross_entropy(p, q):
	return -sum([p[i]*log2(q[i]) for i in range(len(p))])

# define data
p = [0.10, 0.40, 0.50]
q = [0.80, 0.15, 0.05]
# calculate cross entropy H(P, Q)
ce_pq = cross_entropy(p, q)
print('H(P, Q): %.3f bits' % ce_pq)
# calculate cross entropy H(Q, P)
ce_qp = cross_entropy(q, p)
print('H(Q, P): %.3f bits' % ce_qp)
```

## Lesson 06: Naive Classifiers


```py
# example of the majority class naive classifier in scikit-learn
from numpy import asarray
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
# define dataset
X = asarray([0 for _ in range(100)])
class0 = [0 for _ in range(25)]
class1 = [1 for _ in range(75)]
y = asarray(class0 + class1)
# reshape data for sklearn
X = X.reshape((len(X), 1))
# define model
model = DummyClassifier(strategy='most_frequent')
# fit model
model.fit(X, y)
# make predictions
yhat = model.predict(X)
# calculate accuracy
accuracy = accuracy_score(y, yhat)
print('Accuracy: %.3f' % accuracy)
```

## Lesson 07: Probability Scores

### Log Loss Score

```py
# example of log loss
from numpy import asarray
from sklearn.metrics import log_loss
# define data
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# define data as expected, e.g. probability for each event {0, 1}
y_true = asarray([[v, 1-v] for v in y_true])
y_pred = asarray([[v, 1-v] for v in y_pred])
# calculate log loss
loss = log_loss(y_true, y_pred)
print(loss)
```

### Brier Score
 
```py
# example of brier loss
from sklearn.metrics import brier_score_loss
# define data
y_true = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
y_pred = [0.8, 0.9, 0.9, 0.6, 0.8, 0.1, 0.4, 0.2, 0.1, 0.3]
# calculate brier score
score = brier_score_loss(y_true, y_pred, pos_label=1)
print(score)
```
