## Embed to Control

This is re-implementation of "[Embed to Control: A Locally Linear Latent Dynamics Model for Control from Raw Images](https://arxiv.org/abs/1506.07365)", NIPS, 2015.
known as E2C. We tested on the pendulum environment. For details please refer to our [report](./report.pdf). 


E2C code is from [E2C pytorhc implementation](https://github.com/tung-nd/E2C-pytorch) and the changes are as follows:

- We changed the consistency loss term. There is a typo in the original paper.
- We changed the code of generating the images.

iLQR code is from - [ilqr_pendulum](https://github.com/ipab-rad/ilqr_pendulum) and the changes are as follows:

- We integrated the E2C model.
- We added anlaysis [plots](./evaluate_saved_model.ipynb).

This is joint work with [Kihong-cyber](https://github.com/Kihong-cyber).

