## Machine Learning
1. What is bias?
Bias is error due to overly simplistic assumption. I can lead to the model underfitting your data, making it hard to generalize the knowledge.

2. What is variance?
Variance is error due to too much complexity in the learning algorithm. THis leads to the model to overfit the data. The model is learning too much noise from the training data. 

3. What is the trade off between bias and variance?
Essentially, if you make the model more complex and add more variables, you’ll lose bias but gain some variance—in order to get the optimally reduced amount of error, you’ll have to trade off bias and variance. You don’t want high bias or high variance in your model.

4. What to do to reduce overfitting?
- More training data
- Less complex models
- Regularization techniques(L1, L2, Dropout, early stopping)
- Reduce feature sets. 

5. What to do to reduce underfitting?
- Increase model complexity
- Add more features
- Decrease regularization
- Train longer

6. Advantages and disadvantages of Support Vector Machine (SVM).
Imagine you have a big pile of red and blue marbles mixed together on the ground, and your job is to draw a line between the red and blue marbles so that all the reds are on one side and all the blues are on the other. But you want to make sure that the line you draw is the best line possible, meaning it keeps the two groups as far apart as it can. Linear Kernel: Just a straight line—perfect when the marbles are already somewhat separated. Polynomial Kernel: Like drawing wavy or curved lines; it’s like bending the space around to make separation easier. Radial Basis Function (RBF) or Gaussian Kernel: It’s like putting a stretchy sheet around marbles so you can separate even the trickiest mixed patterns.
- Advantages: Effective in high-dimension spaces, robust to overfitting, 
- Disadvantages: computationally intensive with large datasets. Less effective on noise date, difficult inpterpretation.

7. Advantages and disadvantages of Random Forest (RF).
- Advantanges: Robust to overfitting, handles high dimensional data, robust to noise and outlier, handles missing values, versatil for classification and regression. 
- Disadvantages: Interpretability, computationally intensive, slower inference. 

8. What are precision, recall and F1 score?
- Precision: Out of all the positive predictions, how many are actually correct. TP/(TP+FP)
- Recall: Out of all the actual positvies, how many did the model correctly identify. TP/(TP+FN)
- F1: A single metric that balances both precision and recall. (2 x Precision x recall)/(Precision + Recall)

9. Explain what is boost and bagging
Both are emsemble learning techniques that combine multiple modelsto improve overall performance:
- Bagging: Random subsets of the training data are created with replacement, meaning some data points maybe repleated in each subset while others maybe excluded. Each subset is used to train an independent model. The predictions from each model are conbined. 
- Boosting: Models are trained one after another. The first model makes predictions, and its errors are identified. The next model is trained with a focus on the instances that had high errors by the previous models. Adjusting the weights of misclassified instances, so the model pays more attention. Final prediction combine all models, with each model's contribution weighted.

10. What is dropout?
Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data.

11. How do you handle missing data?
Either drop it out or replace it a new value. 

12. Differences between Type 1 and Type 2 error.
Type 1 error (False Positive). Type 2 error (False Negative). Type I error, or a false positive, would be telling a man he was pregnant, while Type II error would be telling a pregnant woman she wasn’t.

13. How would you handle an imbalanced dataset?
- Collect more data to even the imbalances.
- Resample the dataset to correct for imbalances.
- Try a different algorithm on the dat set. (RF by adjusting the calss weights, which makes the algorithm pay more attention to the minority class. As well as using anomaly detection.)

## Natural Language Processing
1. Brief introduce the transformer model
- Self attention
The attention mechanism ia an additional attention layer that enables the model to focus on specific parts of the input while performing a task. It achieves this by dynamically assigning weights to different elements in the input, indicating their relative importance/relevance. It allows the model to process words in parallel, which makes it significantly faster than RNNs. Also it enables to effectively capture long-range dependencies, which is suitable for tasks where long-term context is essential. 
- Multi-head attention
This enables the model to recognise different types of correlations and patterns in the input sequence. Each attention head learns to pay attention to different parts of the input, allowing the model to capture a wide range of characteristics and dependencies.
- Positional Encoding
Transformer model processes input sequences in parallel, so that it lacks the sequenctial information. Postional encoding is applied to the input embeddings to offer the positional information.
2. Brief introduce GPT model


3. Explain the word embeddings.
Word embeddings are defined as the dense, low-dimensional vector representations of words that capture semantic and contextual information about words. The goal of word embeddings is to capture relationships and similarities between words by representing them as dense vectors. It capture the semantic similarity between words. Compared to one-shot encoding, it reduced the dimensionality of word representations. Words embeddings have a fixed length and represent words as dense vectors. 

4. What is Word2Vec?
Word2Vec is a shallow neural net. Its input is a text corpuss nad its output is a set of vectors. Word2Vec turns text into a numerical form that DNN can understands. King to Queen as man to woman. 
5. Differences between stemming and lemmaliazation
- Stemming is a process that removes suffixes from words to reduce them to their base or root form. The goal is to strip down words to a common base form, often not a real word. (flies -> fli)
- Lemmatization reduces words to their base or dictionary form (lemma). It involves looking up a word's base form in a dictionary and ensuring that the result is a real word. (flies -> fly)
6. Explain the cosine similarity and its importance
Cosine similarity is used to compare two vectors that represent text. 
7. Limitations of a standard RNN. 
Vanishing gradient problem. Gradients decrease exponenetially as they progapate backwards through time. Because of this, it is difficult to capture long-term dependencies during training. Similiar to exploding gradient problems, in which gradients get exceedingly big and cause unstable training. 
8. What is a long short-term memory (LSTM) network?
A long term momory network is a type of recurrent neural network (RNN) that is designed to solve vanish and exploding problems. It contains three gates: input gate, forget gate and output gate. The input gate controls how much new information should be stored in the memory cell. The forget gate determines which information from the memory cell should be destroyed or forgotten. The output gate controls how much information is output from the memory cell to the next time step. These gates are controlled by activation functions, which are commonly sigmoid and tanh functions, and allow the LSTM to selectively update, forget, and output data from the memory cell.
9. What is Masked Language Model?
Masked language models help learners to understand deep representations in downstream tasks by taking an output from the corrupt input. This model is often used to predict the words to be used in a sentence. 

