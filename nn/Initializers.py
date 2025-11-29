import numpy as np
from jaxtyping import Float


class Init:
    he = "he"
    xavier = "xavier"
    as_arr = [he, xavier]


#Initializes weights using He initialization.
def init_weights(in_features, out_features, method=Init.he) -> Float[np.ndarray, "in_feat out_feat"]:
    if method == Init.he:
        scale = np.sqrt(2.0 / in_features)
    elif method == Init.xavier:
        scale = np.sqrt(2.0 / (in_features + out_features))

    else:
        scale = 0.01
    return np.random.randn(in_features, out_features) * scale
