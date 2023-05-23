# ML_Gender_Wage_GAP
ML application in public policy research 

Most empirical economic research focuses on questions of
causality. However, machine learning methods can actually be used
in economic research or policy making when the goal is prediction.

[[mlKLMO15](#id13)] is a short paper which makes this point.

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
us solve them more effectively.  [[mlKLMO15](#id13)].


One of their examples is the allocation of joint replacements for
osteoarthritis in elderly patients. Joint replacements are costly,
both monetarily and in terms of potentially painful recovery from
surgery. Joint replacements may not be worthwhile for patients who do
not live long enough afterward to enjoy the
benefits. [[mlKLMO15](#id13)] uses machine learning methods to
predict mortality and argues that avoiding joint replacements
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
  trial to help decide whether to offer bail [[mlKLL+17](#id12)].  


We investigated one such prediction policy problem in
[recidivism](https://datascience.quantecon.org/recidivism.html).
