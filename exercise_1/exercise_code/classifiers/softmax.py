"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier


def cross_entropy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    
    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    loss = 0.0
    dW = np.zeros_like(W)
    
    N = X.shape[0]
    D = W.shape[0]
    C = W.shape[1]
    
    for i in range(N) :
        f = X[i,:].dot(W)
        f -= np.max(f)
        p = np.exp(f) / np.sum(np.exp(f))
        loss += - np.log(p[y[i]])
        
        for c in range(C) :
            for d in range(D) :
                dW[d,c] += (p[c] - (c == y[i])) * X[i,d]
    dW /= N
    dW += reg * W

    loss /= N
    loss += 0.5 * reg * np.sum(W * W)
 
    pass

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.
    
    Inputs and outputs are the same as in cross_entropy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    
    N = X.shape[0]
    C = W.shape[1]
    
    F = np.dot(X,W)
    F -= np.max(F, axis = 1, keepdims = True)
    P = np.exp(F) / np.sum(np.exp(F), axis = 1, keepdims = True)

    Y = np.zeros_like(P)
    
    Y[np.arange(N),y.astype(int)] += 1

    loss += np.sum(- np.log(P[np.arange(N), y.astype(int)]))
    loss /= N
    loss += 0.5 * reg * np.sum(W**2)

    dW = np.dot((X.T),(P-Y))
    dW /= N
    dW += reg * W

    pass

    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [2e-7, 3e-7, 4e-7, 5e-7]
    regularization_strengths = [1e4, 2e4, 3e4, 4e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   # 
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################
    
    for i in learning_rates:
        for j in regularization_strengths:
            SC = SoftmaxClassifier()
            SC.train(X_train, y_train, learning_rate=i, reg=j, num_iters=400, batch_size=200, verbose=False)
            key = (i,j)
            val = np.mean(y_train == SC.predict(X_train)),np.mean(y_val == SC.predict(X_val))
            results[key] = val
            classifier = [SC, np.mean(y_train == SC.predict(X_train)),np.mean(y_val == SC.predict(X_val))]
            all_classifiers.append(classifier)
            if best_val < np.mean(y_val == SC.predict(X_val)) :
                best_val = np.mean(y_val == SC.predict(X_val))
                best_softmax = SC
                
    print(all_classifiers[-1][0])
    pass

    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################
        
    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))
        
    print('best validation accuracy achieved during validation: %f' % best_val)

    return best_softmax, results, all_classifiers
