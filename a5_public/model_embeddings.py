#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        self.embed_char_size = 50
        self.embed_size = embed_size
        char_vocab_size = len(vocab.char2id)
        self.cnn = CNN(self.embed_char_size, self.embed_size)
        self.highway = Highway(self.embed_size)
        self.drop_out = nn.Dropout(0.3)
        self.char_embedding = nn.Embedding(char_vocab_size, self.embed_char_size, padding_idx=vocab.char2id['<pad>'])
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        char_embeddings = self.char_embedding(input)
        sentence_length, batch_size, max_word_length, embed_char = char_embeddings.shape
        char_embeddings_reshaped = char_embeddings.reshape((sentence_length * batch_size, max_word_length, embed_char))
        char_embeddings_reshaped = char_embeddings_reshaped.permute(0, 2, 1)
        conv_out = self.cnn(char_embeddings_reshaped)
        highway = self.highway(conv_out)
        return self.drop_out(highway).reshape(sentence_length, batch_size, self.embed_size)
        ### END YOUR CODE

