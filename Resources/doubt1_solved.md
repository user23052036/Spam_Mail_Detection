
You printed:

```python
print(X_train.iloc[0])
```

Output:

```
Don know. I did't msg him recently.
```

Then you printed:

```python
print(X_train[0])
```

Output:

```
Go until jurong point, crazy..
```

Now you’re confused why these are different.

---

# 🔥 Here’s Exactly What’s Happening

`X_train` is a **pandas Series**.

In pandas, there are TWO ways to access elements:

### 1️⃣ `.iloc[index]`

Access by **position**

```python
X_train.iloc[0]
```

→ first row in the training set (after shuffle)

---

### 2️⃣ `[index]`

Access by **label**

```python
X_train[0]
```

This does **NOT** mean “first row”.

It means:

> Give me the row whose index label is 0.

After `train_test_split`, pandas keeps original indices.

So your training set might look like this:

```
index    message
102      ...
445      ...
0        Go until jurong point...
3312     ...
```

So:

* `.iloc[0]` → first row in training set → “Don know. I did't msg him recently.”
* `[0]` → row whose original index was 0 → “Go until jurong point…”

These are DIFFERENT rows.

That’s why your confusion happened.

---

# 🎯 The Key Concept

Pandas keeps original index labels after splitting.

So your training set has shuffled rows but original indices.

Therefore:

| Code      | Meaning                                      |
| --------- | -------------------------------------------- |
| `iloc[0]` | First row in training set                    |
| `[0]`     | Row with label index 0 from original dataset |

---

# ✅ Why Your TF-IDF Matches `iloc[0]`

Earlier we saw TF-IDF row 0 had words:

```
don
know
did
msg
recently
```

That matches:

```
Don know. I did't msg him recently.
```

So everything is consistent.

Your matrix row 0 corresponds to:

```python
X_train.iloc[0]
```

Not `X_train[0]`.

---

# 🧠 Why This Is Important

This confusion happens because:

* NumPy arrays are position-based.
* Pandas Series can be label-based OR position-based.

And you're mixing the two mental models.

Sparse matrix row 0 = first row passed into vectorizer.

Which was:

```python
X_train.iloc[0]
```

---

# 🛠 If You Want Clean Indexing

You can reset indices after split:

```python
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
```

Then:

* `X_train[0]`
* `X_train.iloc[0]`

will return the same thing.

---

# 💡 Final Clarity

Nothing is broken.
Nothing is inconsistent.
Nothing is wrong.

You just encountered:

> The difference between label-based indexing and position-based indexing in pandas.

That’s all.

---
