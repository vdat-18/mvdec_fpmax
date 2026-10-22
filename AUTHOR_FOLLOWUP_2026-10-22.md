Subject: Re: Regarding the response: Dataset for research on air pollution in your study "Multi-view Deep Embedded Clustering: Exploring a new dimension of air pollution"

Dear Mr. Kassem and co-authors,

Thank you again for the air-pollution file you sent in April 2025. I have one more short question, this time about Table 2 (public datasets), which I am trying to reproduce.

With DEKM_F as View1 and the 10-D bottleneck U-Net as View2, I find that the fused result depends strongly on the random seed. On REUTERS-10K, 10 seeds of the joint model give a fused ACC between 0.61 and 0.78 (mean 0.71), and on RCV1-10K between 0.60 and 0.76 (mean 0.67). Your fused rows (78.13 on REUTERS, 71.93 on RCV1) lie inside these ranges, but your fused NMI (61.41 on REUTERS) and View2 row (65.01 / 51.29) I cannot reach with any seed.

1. For the "Proposed method" rows in Table 2, how many runs/seeds were made, and how was the reported run chosen (average, best ACC, best NMI, or a checkpoint)?
2. In the joint model, are the two views trained from scratch together, or is View1 initialised from a pretrained DEKM_F, and is anything besides Eq. 4 and Eq. 11 used to align the two latent spaces?

A one-line answer to either question, or the training script for the REUTERS row, would help a lot. I can share my code, seeds and logs if useful.

Thank you very much for your time.

Best regards,
Dat Nguyen-Van
ngvdat.w@gmail.com
