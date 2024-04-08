# Special Cases

## Mixture of Experts error rates
Due to the Top-K gating performed in the hidden layer of Mixture of Experts models, small errors can be amplified 
greatly in cases where a different expert is selected, which leads to a higher than normal variance in the error rate
of the final logits. In testing done on Mixtral running in half precision, the standard deviation of the absolute error 
rate of the logits compared to those from the default model was found to be around 2e-3.

There are two main ways to mitigate this:
1. Disable preprocessing options by using `HookedTransformer.from_pretrained_no_processing` instead of `HookedTransformer.from_pretrained`
2. Increase the precision of the data type used in the model
