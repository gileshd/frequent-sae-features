# Investigating Frequent SAE Features

## Compare the Principal Components

1. Get trained SAE
2. Collect dataset of activations
    - This is actually a little tricky because memory requirements are v big.
3. Compute PCA of activations
4. Calculate highly active features
5. Compare decoding vectors of these features with the PCA components
    - Do they directly correspond to the PCA components?
    - Are they orthogonal to each other?
        - More so than less frequent features?
    - How does the subspace spanned by these features compare to the top-k PC subspace?


## Captain's Log

### Session 1

- Load in GPT-2 small
- Load in the SAE

## Session 2

