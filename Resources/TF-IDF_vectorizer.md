
# TF-IDF → Logistic Regression

### How TF-IDF shapes gradient flow and decision boundary geometry

---

## Step 1 — Start from Logistic Regression

Logistic regression computes:

$$
z = w^\top x + b
$$

$$
\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Logistic regression does **not** understand text.

It only understands:

* A feature vector $x$
* A weight vector $w$
* A dot product $w^\top x$

Everything else (tokenization, TF-IDF, normalization) is preprocessing whose job is to produce a meaningful $x$.

> **Key question:**
> What kind of vector $x$ makes the dot product $w^\top x$ meaningful for text?

TF-IDF answers that.

---

## Step 2 — What the Dot Product Actually Does

$$
w^\top x = \sum_{i=1}^{V} w_i x_i
$$

Each feature contributes additively:

$$
\text{Contribution of word } i = w_i \cdot x_i
$$

* $w_i$ = how predictive word $i$ is (learned weight)
* $x_i$ = how strongly the document expresses word $i$ (TF-IDF value)

If a word is strong in the document **and** has a strong weight, it pushes the logit strongly.

So TF-IDF’s entire job is to produce per-word intensity values $x_i$ that make those contributions informative.

---

## Step 3 — Why Raw Counts Are Bad (Geometric View)

If $x$ is raw word counts:

* Long documents → large magnitude vectors
* Short documents → small magnitude vectors

Then:

$$
z = w^\top x
$$

becomes influenced by document length.

The classifier might accidentally learn:

> “Long emails = spam”

That’s wrong.

We need:

* Length invariance
* Discriminative word emphasis

TF-IDF gives both.

---

## Step 4 — TF: Local Signal Strength

Term Frequency:

$$
\text{TF}(t,d) = \frac{\text{count}(t \text{ in } d)}{\text{total words in } d}
$$

Properties:

* Measures importance relative to document length
* Keeps features bounded
* Makes gradient scale proportional to relative word dominance

Sublinear TF (optional):

$$
\text{TF} = 1 + \log(\text{count})
$$

This dampens extreme repetition.

---

## Step 5 — IDF: Global Feature Scaling

Smooth IDF:

$$
\text{IDF}(t) = \log\left(\frac{N+1}{\text{df}(t)+1}\right) + 1
$$

If $\text{df}(t) \approx N$, then:

$$
\text{IDF}(t) \approx 0
$$

So the feature shrinks across the dataset.

### Why This Matters for Gradients

Logistic regression gradient:

$$
\frac{\partial L}{\partial w_j}
===============================

\sum_n (\hat{y}*n - y_n) x*{n,j}
$$

If $x_{n,j}$ is nonzero for almost every document but not discriminative:

* Gradients push in conflicting directions
* Weight oscillates
* Model wastes capacity

IDF suppresses such features.

It acts like **variance-based gradient filtering**.

---

## Step 6 — TF × IDF = Signal That Survives Training

A word becomes powerful only if:

1. It appears strongly in this document (high TF)
2. It appears rarely across documents (high IDF)

Thus TF-IDF ensures only discriminative features generate strong gradient updates.

---

## Step 7 — L2 Normalization (Geometry Level)

After TF-IDF, we normalize:

$$
x \leftarrow \frac{x}{|x|_2}
$$

Then:

$$
w^\top x = |w| |x| \cos \theta
$$

Since $|x| = 1$:

$$
w^\top x = |w| \cos \theta
$$

Classification now depends only on angle.

Not magnitude.

Logistic regression becomes an **angular separator**.

Documents are separated by orientation in feature space.

---

# Part 1 — Gradient Dynamics

Logistic loss:

$$
L = -\sum_n \left[
y_n \log \hat{y}_n
+
(1-y_n)\log(1-\hat{y}_n)
\right]
$$

with:

$$
\hat{y}_n = \sigma(w^\top x_n)
$$

Gradient:

$$
\frac{\partial L}{\partial w_j}
===============================

\sum_n (\hat{y}*n - y_n) x*{n,j}
$$

Key insight:

> Feature value $x_{n,j}$ directly scales gradient magnitude.

### Without TF-IDF

* Large noisy gradients from common words
* Instability

### With IDF

* Suppresses globally common features
* Reduces gradient noise

### With TF

* Prevents long documents from dominating updates

### With L2 normalization

* Gradient magnitude depends only on prediction error
* Optimization stabilizes

---

# Part 2 — Decision Boundary Geometry

Decision boundary:

$$
w^\top x + b = 0
$$

This is a hyperplane in $\mathbb{R}^V$.

Effects of TF-IDF:

1. IDF rescales axes (rare words stretched, common words compressed)
2. L2 normalization projects points onto unit sphere
3. Spam and ham become directional clusters

Logistic regression finds a hyperplane separating those directions.

Intuition:

$$
w \propto (\text{spam centroid}) - (\text{ham centroid})
$$

The hyperplane is perpendicular to that difference.

---

# Why Sublinear TF Helps

If a spam email repeats:

"free free free free free"

Raw TF → linear gradient growth.

Sublinear TF:

$$
\text{TF} = 1 + \log(\text{count})
$$

Growth slows.

Repetition influence saturates.

It behaves like soft gradient clipping.

---
