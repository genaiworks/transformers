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
























Ref:
https://www.youtube.com/watch?v=bCz4OMemCcA





