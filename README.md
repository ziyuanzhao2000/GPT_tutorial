### GPT tutorial

In this exercise, I followed Andrew Karpathy's excellent [YouTube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) to train a (decoder-only) GPT language model on the entire corpus of Shakespeare's works. Then I asked the model to generate text without further conditioning or fine-tuning. In the scaled-up configuration (4 heads, 4 self-attention layers, 256 characters long attention window, and a token embedding size of 384), the output is pretty funny, and looks like Shakespeare, but is still mostly gibberish.

To take it one step further I rewrote the data generation code to repurpose the GPT model for answering simple integer addition problems like $12+34=46$. The two tricks based on Andrew's suggestions are 

1. When training the model I invert the digits for the sum so the model is forced to predict the least significant digits first based on the context (problem statement), this is to mimic how humans do addition; 
1. When computing the loss function, I masked the part of the problem statement because it's irrelevant whether the model can guess that or not.

Even with a simple configuration, my model does learn to add double-digit numbers after some training. It's quite interesting that it seems always to get the hundredth right but sometimes is off by one or two in the tenth and the last place. For example, in this example $92+41$ the model predicts $124$ whereas it should be $133$.