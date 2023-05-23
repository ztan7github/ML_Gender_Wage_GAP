# ML_Gender_Wage_GAP
ML application in public policy research 

Most empirical research focuses on questions of
causality. However, machine learning methods can actually be used
in economic research or policy making when the goal is prediction.

> Consider two toy examples. One policymaker facing a drought must
decide whether to invest in a rain dance to increase the chance
of rain.  Another seeing clouds must decide whether to take an
umbrella to work to avoid getting wet on the way home. Both
decisions could benefit from an empirical study of rain. But each
has differ-ent requirements of the estimator. One requires
causality: Do rain dances cause rain? The other does not, needing
only prediction: Is the chance of rain high enough to merit an
umbrella?  We often focus on rain dance–like policy problems. But
there are also many umbrella-like policy problems.  Not only are
these prediction problems neglected, machine learning can help
us solve them more effectively.


One of their examples is the allocation of joint replacements for
osteoarthritis in elderly patients. Joint replacements are costly,
both monetarily and in terms of potentially painful recovery from
surgery. Joint replacements may not be worthwhile for patients who do
not live long enough afterward to enjoy the
benefits. Previous research used machine learning methods to
predict mortality and argued that avoiding joint replacements
for people with the highest predicted mortality risk could lead to
sizable benefits.

Other situations where improved prediction could improve economic
policy include:

- Targeting safety or health inspections.  
- Predicting highest risk youth for targeting interventions.  
- Improved risk scoring in insurance markets to reduce adverse
  selection.  
- Improved credit scoring to better allocate credit.  
- Predicting the risk someone accused of a crime does not show up for
  trial to help decide whether to offer bail.

### Partially Linear Regression

To be more concrete, consider a regression model.  We have some
regressor of interest, $ d $, and we want to estimate the effect of $ d $
on $ y $. We have a rich enough set of controls $ x $ that we are willing to
believe that $ E[\epsilon|d,x] = 0 $ . $ d_i $ and $ y_i $ are scalars, while
$ x_i $ is a vector. We are not interested in $ x $ per se, but we need to
include it to avoid omitted variable bias. Suppose the true model
generating the data is:

$$
y = \theta d + f(x) + \epsilon
$$

where $ f(x) $ is some unknown function. This is called a
partially linear model: linear in $ d $, but not in
$ x $ .

A typical applied econometric approach for this model would
be to choose some transform of $ x $, say $ X = T(x) $, where $ X $
could be some subset of $ x $ , perhaps along with interactions, powers, and
so on. Then, we estimate a linear regression,

$$
y = \theta d + X'\beta + e
$$

and then perhaps also report results for a handful of different
choices of $ T(x) $ .

Some downsides to the typical applied econometric practice
include:

- The choice of $ T $ is arbitrary, which opens the door to specification
  searching and p-hacking.  
- If $ x $ is high dimensional and $ X $ is low dimensional, a poor
  choice will lead to omitted variable bias. Even if $ x $ is low
  dimensional,  omitted variable bias occurs if $ f(x) $ is poorly approximated by $ X'\beta $.  


In some sense, machine learning can be thought of as a way to
choose $ T $ in an automated and data-driven way. Choosing which machine learning method
to use and tuning parameters specifically for that method are still potentially arbitrary
decisions, but these decisions may have less impact.

Economic researchers typically don’t just want an estimate of
$ \theta $, $ \hat{\theta} $. Instead, they want to know that
$ \hat{\theta} $ has good statistical properties (it should at
least be consistent), and they want some way to quantify how uncertain is
$ \hat{\theta} $ (i.e. they want a standard error). The complexity
of machine learning methods makes their statistical properties
difficult to understand. If we want $ \hat{\theta} $ to have
known and good statistical properties, we must make sure we use machine
learning methods correctly.  A procedure to estimate
$ \theta $ in the partially linear model is as follows:

1. Predict $ y $ and $ d $ from $ x $ using any machine
  learning method with “cross-fitting”.  
  - Partition the data in $ k $ subsets.  
  - For the $ j $ th subset, train models to predict $ y $ and $ d $
    using the other $ k-1 $ subsets. Denote the predictions from
    these models as $ p^y_{-j}(x) $ and  $ p^d_{-j}(x) $.  
  - For $ y_i $ in the $ j $ -th subset use the other
    $ k-1 $ subsets to predict $ \hat{y}_i = p^y_{-j(i)}(x_i) $  
1. Partial out $ x $ : let $ \tilde{y}_i = y_i - \hat{y}_i $
  and $ \tilde{d}_i = d_i - \hat{d}_i $.  
1. Regress $ \tilde{y}_i $ on $ \tilde{d}_i $, let
  $ \hat{\theta} $ be the estimated coefficient for
  $ \tilde{d}_i $ . $ \hat{\theta} $ is consistent,
  asymptotically normal, and has the usual standard error (i.e. the
  standard error given by statsmodels is correct).  


Some remarks:

- This procedure gives a $ \hat{\theta} $ that has the same
  asymptotic distribution as what we would get if we knew the true
  $ f(x) $ . In statistics, we call this an oracle property,
  because it is as if an all knowing oracle told us $ f(x) $.  
- This procedure requires some technical conditions on the data-generating
  process and machine learning estimator, but we will not worry about them here. See
  [[mlCCD+18](#id14)] for details.  


Here is code implementing the above idea.
