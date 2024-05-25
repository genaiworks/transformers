Attention of ALL you Need

RNN: 
    Allows mapping one seq of input to another seq of output.
    Hidden state of previous time step along with word in the input sent to the next word to produce output.
    Problem:
        They are slow for long sequences.  We have for loop where we do same operation for every token in the sequence.
        Vanishing or exploding gradient: 
            Pytorch converts network into computational graph.   
            Pytorch calculates the derivates to the loss function w.r.t each weights.  Chain Rule: Longer the chain of compuation resulting number are lower at each step (.5 * .5)... If we have very long chain of computation... compuation of number become very small.
        Difficulty of accessing information from long time ago.
Encoder
    Input ---> Embedding ---> Main Transformer Block
    Transformer Block:
        Input Embedding --->Split into V, K, Q ---> MultiHeaded Attention --->Normalization ---> Feed Forward Network ---> Normalization
        Input Embedding  --->Skip Connection Input --->SKIP MultiHeaded Attention--->Normalization ---> SKIP Feed Forward Network ---> Normalization


Nx ===> Decoder and Encoder blocks are repeated many times.

Masked MultiHeaded Attention:
    Main advantage of the attention is all operation can occur in parallel which is in constrast to the sequence to sequence model/LSTM/GRU/RNN where operation occurs in sequence.
    We mask the input to the decoder so that the first output only has access to the first element, second output only has access to first two input to the decoder and so on.

Embedding:
    Original Sentence                                   This    is      Awesome
    Input ID                                            108     9       803
    position of word in the vocabulary
    Embedding (Map each ID to 512 number Vector embeddings)

    The 512 integers Vector in embedding for EACH word is NOT fixed, our model is learned and changed the numbers of vector integer in the embedding is changed in such a way that it represent the meaning of the sentence.  
    Input IDs will NOT change since the embedding will not change.
    BUT embeddings are changed as model learn the meaning of the sentence.
    Each word of the vocabulary is mapped to embedding of size 512.

Positional Encoding: 
    Transformer block is permutationally invariant, transformer is invariant to the order of words in a sentence.
    Positional encoding is embeddings added to the input AND output embedding so that it is aware of the position of word in a sentence.
    Each word should carry information about its position in the sentence. Embeddings does not convey any infromation about where that word is the sentence.
    We want model to treat words that are close to each other as close and that are distant to each other as distant.
    We need to tell model how words are distributed inside the sentence.
    Positional encoding represent a pattern that can be learned by the model.

    STEP 1:
        Convert the input sentence to the embeddings of size 512
        Original Sentence                                   This    is      Awesome
    STEP 2:
        Then we create 512 Vector size position embedding and add this position encoding vector to the embedding vector.  Position encoding is created only once and used in every sentence during training and inference.
        This POSITIONAL ENCODING vector is NOT learned during the training process.  This positional encoding represent position of word inside the sentence.
    STEP 3:
        Sum 512 size input embedding vector to 512 size positional embedding vector.
    STEP 4:
        Even position encoding vector is created using sin position encoding
        Odd postion encoding vected is created using cos position encoding
        cos and sin are used because they  Naturally represent a pattern that the model can recognize as continusoud so relative positon are easy to see for the model.

Self Attention With Single Head:
    Self attention allows model to related words to each other.
    Input Embeddings: captures the meaning of the words.
    Positional Encoding: Gives information of position of the words inside the sentence.
    We want Self Attention to RELATE words to each other.

    Input of 3 words with the dimension of 512, each word represented by vector of size 512.
    Original Sentence                            This    is      Awesome
    The matrices Q, K and V are just input sentence:
    To Calcularte Self Attention = Attention(Q, K, V) = softmax(Q, transpose(K)/sqrt(d))V

    Q = matrix (3,512)
    KT (K Transpose) = matrix (512,3)
    Q*K = matrix (3,3)
    softmax(Q*K/sqrt(512)) ===>Here we have dot product of first row with first col etc== Return 3x3 matrix==> Where each cell in the matrix represent relation of word with other word.  Multipying with Softmax will make all the values in the ROW sum to 1.  Dot product of each word with every other word in the sentence.  Dot product of each word with embedding of every other word in the sentence.  This dot product represent the score how intense relationship of one word to another.

    softmax(Q*K/sqrt(512)) * V (6x512) = Attention (3 * 512)
    Here each row in the matrix captures not only the meaning or positon in the sentence but also each words interaction with each other.  Here we get 3 words each with embedding of 512 columns.
    This 512 vector embedding captures the relationship of word with all the other words in the sentence.

Self Attention Properties:
    1. Self attention is permutation invariant.  If we do not consider the contribution of the positional encoding.
    2. Value of embedding of each word related to other words does not change if the order of word changes.
    3. Self attention requires no parameters.  Up to now the interaction between words have been driven by their embedding and the positional encoding.  This will change later.
    4. Each value in the diagonal is the highest becasue of more relationship with the self.
    5. If we do not want some positon to interact we can replace their value to -infinity before applying softmax in this matrix and the model will learn those interactions.

MultiHead Attention:
    Single Head = softmax(Q*K/sqrt(512)) * V (6x512) = Attention (3 * 512)
    Multihead(Q,K,V) = concat (head1, head2,... head)W
    head(i) = Attention(QWi,KWi,VWi)

    Take the Input Matrix and Make 3 copies of it, Multiply each matrix copy by 3 parameter matrix Wq, Wk, Wv
    1. Query Q Matrix (Seq x Dmodel) * Wq (Dmodel x Dmodel) == (Seq x Dmodel) size matrix
        Split this matrix into dmodel dimension for each word ==> Q1 (first word) Q2 (second word) Q3 (third word) ...

    2. Key K Matrix (Seq x Dmodel) * Wk  (Dmodel x Dmodel) == (Seq x Dmodel) size matrix
        Split this matrix into dmodel dimension for each word ==> K1 (first word) K2 (second word) K3 (third word) ...

    3. Value V Matrix (Seq x Dmodel) * Wv (Dmodel x Dmodel) ==(Seq x Dmodel) size matrix
        Split this matrix into dmodel dimension for each word ==> V1 (first word) V2 (second word) V3 (third word) ...

    Every head will see the full sentence but smaller part of embedding of each word.  

    Q1 K1 V1 ===> HEAD 1 (seq h * d)
    Q2 K2 V2 ===> HEAD 2
    Q3 K3 V3 ===> HEAD 3

    H = concat (head1, head2,... head) 
    Each head will montior the different aspect of each word in the sentence.  One word may be verb in some case noun in some case it will learn all the contexts of the words in the sentence.  One head may learn that word as noun, another head may learn that word as verb etc.

    Query 
    Keys 
    Values
    K/V similar to Python dictionary.

Add/Norm (Layer Normalization):
    Here we normalize the data, where we calcuate mean and variance of each item of batch independently of each other.  Normalizing so that all values are between 0 and 1.
    modified val = x - mean / sqrt(sq(stdev)) --> Value between 0 and 1
    We also introduce two new parameters usually called gammat (mulitplictive)
    and beta (additive) that added some fluctuations in the data because all value between 0 and 1 may be too restrictive. Network learns to tune these two parameters.


Decoder
    Output of Encoder is send as V, K as input to decoder---> Decoder MultiHeaded Attention 
    Query Q goes directly from Masked MultiHeaded Attention to Multi Headed Attention
    Decoder Block:
        On Top we have SAME Transformer Block as used in encoder.
        Here 2 Inputs we have K, V from previous part of Encoder
        And Q input from previous part of Decoder

    Decoder Block = Transformer Block similar to Encoder + Masked MultiHeaded Attention + Normalization

    Embedding from Ouput ---> Split into V, K, Q ---> Masked MultiHeaded Attention ---> Normalization ---> Transformer Block

    Masked MultiHeaded Attention:
    Goal is to make model causal.  It means the ouput at a certain position can only depends on the words on the previous position.  The model must NOT see next word.

    The words that we want to hide we delete those values and put minus infinity in place of values that we want to hide.  Then we apply Softmax.
    So that Softmax will replace these values by zero.  We want to replace all the values above the diagnonal to minus infinity.

    




























Ref:
https://www.youtube.com/watch?v=bCz4OMemCcA


Learning: parameter estimation using differentiation and gradient descent.
fit: Means make algorithm learn from data.
take data --> choose a model --> estimate the parameters of the model => good prediction on new data
1. Given input data and outputs as well as weights 
2. Measure loss: By computing the resulting output with the ground truth.
3. To Optimize parameters/weights of the model:
    Change in error following unit change in weights or gradient of error w.r.t parameters is calculated using the chain rule.
4. The value of the weights is updated in the direction that leads to decrease in error.
5. Repeat till error on unseen data falls below acceptable level.

FORWARD PASS
Input Data
Estimate weight and bias parameters in the model based on the data we have.
Inputs + Weight ---> Model With Hidden Layers ---> Output---> Actual Ouputput Given current weights ---> Loss (Desiered Ouput - Actual Output)

BACKWARD PASS
Calculate Gradient Using Chain Rule (find change in loss due to small change in weight)
Change weight to decrease error/loss function

Repeat 

We have model with some unknown parameters we need to estimate those parameters so taht error between some predicted output and actual output is as low as possible.

Find w and b so that loss is at minimum level.

Loss Function/Cost Function:
    Is a function that computes a single numerical value that the learning process attempt to minimize.
    MSE:
    def loss_fn(act, comp):
        sq_diff = (act - comp)**2
        return sq_diff.mean()
    
    def model(t_u, w, b):
        return w * t_u + b

    w = torch.ones(1)
    b = torch.zeros(0)
    t_u = input

    t_c= expected output
    t_p = model(t_u, w, b)
    loss = loss_fn(t_c, t_p)

    How to estimate w and b such that loss is minimum.

Gradient Descent:
    Optimize Loss w.r.t parameters using gradient descent algorithm.
    Compute the rate of change of loss w.r.t each parameter and apply a change to each parameter in the direction of decreasing loss.
    Small change in w and b an d see how much loss is changing in that neighbourhood.
    A unit change in w leads to some change in loss.
    If change is -ve we need to increase w to minimize loss
    If change is +ve we need to decease w to minimize loss.

    Scale the rate of change by very small factor: This scaling factor is called learning rate.
    lr = 1 e-2
    Basic parameter update step for gradient descent.

    w = w - lr * loss_rate_of_change_w

    b = b - lr * loss_rate_of_change_b

    Chain Rule:
        To compute the derivative of the loss w.r.t parametr we apply chain rule.
        Derivate of loss w.r.t its input * derivate of the model w.r.t the parameters
        d_loss_fn/d_w = (d_loss_fn/d_t_p) * (d_t_p/d_w)

        Derivate of loss w.r.t weights

Trainining Loop:
    Start with tentative value of paramerters
    Apply update to it for fixed number of intervalue
    Until w and b stop changing

    Epoch: 
        A training iteration for which we update the parameters for all training samples is called an epoch.
    
    def training_loop(n_epochs, lr, params, t_u, t_c):
        for epoch in range(1, n_epochs +1):
            w, b = params

            t_p = model(t_u, w, b) # Forward Pass
            loss = loss_fn(t_p, t_c)

            grad = grad_fn(t_u, t_c, t_p, w_b) # backward pass
            params = params - lr * grad

        return params

Adaptive LR:
    The normalization helps you to get the network trained.  Normalization is easy and effiective tool to import model convergence.

Autograd/Gradient of expression w.r.t input automatically:
    Autograd alllow Pytorch tensors can remember where they come from in terms of the oeprations and parent tensors that originatted them.
    They provide chain of derivate of such operation w.r.t inputs automatically.

params = torch.tensors([1.0,0.0], requires_grad=True)
require_grad=True telling Pytorch to track entire family tree of tensors from operations on params.

loss = loss_fn (model(t_u, *params), t_c)
loss.backward()
params.grad

grad attribute of the param contains the derivate of the loss w.r.t each element of the params

Pytorch computes derivate of the loss throuhout the chain of function and accumulate their values in the grad attribute of those tensors.

Calling backward leads derivates to accumulate at leaf nodes.  We need to zero gradient after explicitly using it for parameter updates.

def training_loop(n_epochs, lr, params, t_u, t_c):
    for epoch in range(1, n_epochs +1):
        if params.grad is not None:
            params.grad.zero_()
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()
        params = (params - lr * params.grad).detach().requires_grad_()

detach().requires_grad_():
    We have build compuational graph, we detach the new param tensor from the compuational graph associated with the updated expression by calling detach().  This way param looses the memory of operation that generated it.
    detach() allows us to release the memory held by old version of the params and need to backpropagate through only the current weights.

Optimizer:
    Several optimization and tricks can help convergence specially when models get complicated.
    Every optimizer constructor takes a list of parameter typically with requires_grad to True as first input.  All parameters passed to the optimizer are retained inside the optimizer object so that optimizer can update their values and access their grad attributes.
    Two methods of optimizer:
        zero_grad
        step

    SGD:
        Vanilla gradient descent
        Stochastic comes from the fact that the gradient is typically obtained from averaging over a random subset of all input samples called minibatch.

        The optimizer itself does not know whether loss was calculated on all samples or random subset, the algorithm is same in both the cases.

    The value of params was updated when step was called.  We do not need to touch it ourselves.

    t_p = model(t_u * params)
    loss = loss_fn(t_p, t_c)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    When we call optimizer.step(): optimizer looked into the param.grad and update param by subtracting lr * grad from it.  Before call to backward we need to zero the gradient at every call to the backward.

Adam:
    Is most sophisticated optimizer in which the learning rate is set adaptively.  In addition there is lot less sentivity to the scaling of the parameters.

    params = torch.tensor([1.0,0.0], requires_grad=True)
    lr = 1e-1
    optimizer = optim.Adam([params], lr =lr)
    training_loop(
        n_epochs= 2000,
        optimizer = otpimizer,
        params = params,
        t_u = t_u,
        t_c = t_c
    )

Switch off autograd when we do not need to by using torch.no_grad context.

def training_loop(n_epoch, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs +1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, val_t_c)
        with torch.no_grad():
            val_t_p = model(val_t_u, *params)
            val_loss = loss_fn(val_t_p, val_t_c)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
with torch.no_grad(): Context manager is used to control the auto grad behavior.

Use NonLinear Activation Function As Key Difference From Linear Model:
    Pytorch nn.Module contains NN building blocks.

    Neuron output = Tanh(wx +b)
        Here x is input and w, b are learned parameters
        Tanh is NonLinear function (Activation function)
    output = f(w*x +b)
    Here f is non linear activation function
    Here w is matrix and x is vector

    The activation function has the role of concentrating the ouput of liner operation into a given range.

    Nonlinear allows the overall network to approximate more complex function.

    By joining may linear + activation units in parallel and stacking them one after another leads to mathematical object that is capable of approximating complicated functions.
    Different combination of units responds to input in different ranges and those parameters are easy to optimize thru gradient descent.

    Universal Approximation to estimates it parameters.

Training:
    Find w and b so taht networks carries out a task correctly.  

import torch.nn as nn

linear_model = nn.Linear(1,1)
linear_model(t_un_val)

y = model(x) ==== correct
y = model.forward(x)==== Error

CALLING an instance of nn.Module with set of argument ends up calling a method named forward with the same arguments.
The forward method executes the forward computation.

call() does important chores BEFORE or AFTER calling forward.

def __call__(self, *input, **kwargs):
    for hook in self._forward_pre_hooks.values():
        hook(self, input)
    result = self.forward(*input, **kwargs)
    for hook in self._forward_hooks.values():
        hook_result = hook(self, input, result)
    for hook in self._backward_hooks.values():

    return result

nn.Linear(input, output, whether linear model include bias=True)

unsqueeze()
    Is used to add extra dimension at axis 1
    t_c = torch.tensor(t_c).unsqueeze(1)

linear_model = nn.Linear(1,1)
optimizer = optim.SGD(linear_model.parameters(), lr =1e-1)
replace params with linear_model.paramters()

Input:
t_c =[.5,.8,.9]
t_c = torch.tensor(t_c).unsqueeze(1) #Here we have extra dimension at axis 1

nn.module init constructor returns all flat list of all parameters encountered, so that we can pass it to the optimizer construct.

def training_loop(n_epochs, optimizer, model, loss_fn, t_u_train, t_u_val, t_c_train, t_c_val):
    for epoch in range(1, n_epochs+1):
        t_p_train = model(t_un_train)
        loss_train = loss_fn(t_p_train, t_c_train)
        t_p_val = model(t_un_val)
        loss_val = loss_fn(t_p_val, t_c_val)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()


nn.MSLoss: MeanSquareError

linear_model = nn.Linear(1,1)
optimizer = optim.SGD(linear_model.parameters(), lr=1e-2)
training_loop(
    n_epochs = 3000,
    optimizer = optimizer,
    model= linear_model,
    loss_fn = nn.MSELoss(),
    t_u_train = t_un_train,
    t_u_val = t_un_val,
    t_c_train = t_c_train,
    t_c_val = t_c_val
)

Concatenate Modules Using nn.Sequential
    Simple NN:
        first linear layer
        activation function # activation  function commonly referred as hidden layer
        second linear layer

    Concatenate Module Using nn.Module:
        seq_model = nn.Sequential(
            nn.Linear(1,12),
            nn.Tanh(),
            nn.Linear(12,1)
        )
    model.parameters() collects weights and bias from both first and the second linear module 
    model.backward() is called on ALL parameters are populated with grad.
    Optimizer update their values during optimizer.step()


Subclassing nn.Module:
    Subclass nn.Module and define .forward(...)function taht takes input to the module return output.
    Autograd takes cares of backward pass automatically.

    class SubClassModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden_linear = nn.Linear(1,12)
            self.hidden_activation = nn.Tanh()
            self.output_linear = nn.Linear(12,1)
        def forward(self, input):
            hidden_t = self.hidden_linear(input)
            activated_t = self.hidden_activation(hidden_t)
            ouput_t = self.output_linear(activated_t)
            return output_t

Recipes:
    Pytorch algo is respresented as computational graph.
    Gradient==> are considered as slope of the function.
    slope of function = derivative of function w.r.t parameters.
    Pytorch variable is node in compuational graph which stores data and gradients.
    Update weights using gradient descent/

Word2Vec and GloVe are known framework to execute word embeddings
    embed = nn.Embedding(2,5) # 2nd word in vocab, 5 dimensional embedding


params = torch.tensor([1.0,0.0])
nepochs = 5000
lr = 1e-2
for epoch in range(nepochs):
    #forward
    w,b =params
    t_p =model(t_un,w,b)
    loss = loss_fn(t_p,t_c)
    #backward pass gradient
    grad = grad_fn(t_un,t_c.t_p,w,b)
    params = params - lr *grad
return params

for epoch in range(nepochs):
    #forward pass
    t_p = model(t_un, *params)
    loss = loss_fn(t_p, t_c)
    if params.gred is not None:
        params.grad.zero_()
    loss.backward()
    params = (params - lr * params.grad).detach().requires_grad_()
    return params

optimizer = optim.Adam([params], lr=lr)
for epoch in range(nepochs):
    #forward pass
    t_p = model(t_un, *params)
    loss = loss_fn(t_p, t_c)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()





































































