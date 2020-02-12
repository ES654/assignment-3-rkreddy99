
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    acc = 0
    for i in range(y_hat.size):
        if y_hat[i]==y[i]:
            acc+=1
    acc = acc/y.size
    return acc

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    q=0
    w=0
    y = list(y)
    y_hat = list(y_hat)

    for i in range(len(y)):
        if y_hat[i]==y[i] and y_hat[i]==cls:
            q+=1
        if y_hat[i]==cls:
            w+=1
    pre = q/max(w,1)
    return pre

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    e=0
    r=0
    y = list(y)
    y_hat = list(y_hat)
    for i in range(len(y)):
        if y_hat[i]==y[i] and y[i]==cls:
            e+=1
        if y[i]==cls:
            r+=1
    try:
        rec = e/r
    except:
        rec=0
    return rec

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    rse=0
    y = list(y)
    y_hat = list(y_hat)
    for i in range(len(y)):
        rse+=(y[i] - y_hat[i])**2
    rse = rse/len(y)
    rse = rse**0.5
    return rse

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)
    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    me=0
    y = list(y)
    y_hat = list(y_hat)
    for i in range(len(y)):
        me+=abs(y_hat[i] - y[i])
    me = me/len(y)
    return me
