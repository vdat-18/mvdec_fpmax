# Draft email to the authors (not sent)

The recipient is the corresponding author printed on the paper's first page
(Jusheng Mi, School of Mathematical Sciences, Hebei Normal University). Fill in the
bracketed placeholders (your name and affiliation) before sending. Nothing here has
been sent.

**To:** mijsh@263.net
**Cc:** (optional) the co-author, Hongli Wang, if you find an address
**Subject:** Request for code and datasets: reproducing Intuitive-K-prototypes (Wang and Mi, 2025)

Dear Professor Mi,

I am [your name] at [affiliation]. I am studying your paper "Intuitive-K-prototypes: A
mixed data clustering algorithm with intuitionistic distribution centroid" (Pattern
Recognition, 2025) and want to use Intuitive-K-prototypes as a clustering backbone in my
own work. I implemented the method from the paper and reproduced most of Tables 11-18,
but several results depend on settings the paper does not list. I would be very
grateful if you could share the **source code** and the **preprocessed datasets** you
used, or, failing that, answer the questions below.

**Where my implementation stands.** Using the paper's formulas (Tables 2-9 and the Iris
numerical example are reproduced exactly), the means of both rows (plain and "with
select prototypes") of Tables 11-18 match within one reported standard deviation on all
eight datasets, but only after fitting some parameters to your numbers (details below).
Iris and BCW match to three decimals.

**Questions about settings not given in the paper:**

1. **Weight index beta (Eq. for attribute weights).** Which value did you use for each
   dataset? Beta = 2 (the Table 8 example) does not reproduce Australian, Heart or BCW
   for me; values of about 8, 20 and 50 do.
2. **Parameter mu in Eq. (26).** With Eq. (26) as printed, mu at your recommended values
   collapses clusters on several datasets. Passing `1 - mu` removes the collapse and
   matches the shape of your Figure 5. Is the printed orientation correct, or does the
   implementation use `1 - mu`?
3. **mu for Soybean** (Table 14 lists none), and the **mu values for Heart-5** (Table 18
   lists 0.3; I need about 0.44) **and Chess** (Table 17 lists 0.2; I need about 0.3).
4. **Soybean attributes.** The text says attributes with a single value are removed;
   Table 10 lists 35. I get your results with the 21 non-constant attributes. Is that
   right?
5. **Algorithm 1.** (a) Is the first object x_0 chosen randomly or fixed? (b) Line 18
   does not say which admissible object is taken: the first in file order, a random
   one, or the farthest? (c) Your "with select prototypes" rows have standard deviation
   0.000 on Iris, Ionosphere, BCW and Soybean, so the selection looks deterministic.
   How? On Ionosphere I can only reach your 0.800 / 0.334 / 0.275 with prototypes
   picked by a Chebyshev-distance farthest-point rule and beta near 1, which I cannot
   justify from the paper.
6. **Random-initialisation rows.** The plain rows have very small standard deviations
   (0.000 on Iris, Soybean and Chess). Did you keep the best of several restarts (by
   which criterion), or fix the random seed? How many runs and restarts?
7. **Heart-5.** My accuracy standard deviation is 0.034-0.039 against your 0.018 even
   when the means match. Is there any additional filtering of runs (for example
   discarding runs with small or empty clusters)?
8. **Data.** Please confirm the exact files, row order and normalisation (for example
   min-max to [0, 1] for numerical attributes) behind Table 10, since Algorithm 1 depends
   on row order.

If sharing the code is not possible, the per-dataset parameter table (beta, mu, gamma,
lambda, number of runs and restarts) and the preprocessed datasets would already settle
most of this. I will cite your work and credit your help in any resulting publication.

Thank you very much for your time.

Kind regards,
[your name]
[affiliation, contact]
