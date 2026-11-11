# Interview prep

## 60-second spoken summary

> I analysed a randomized email experiment: 64,000 retail customers split
> between a men's merchandise email, a women's email and no email. The
> question was how to use the two emails, and whether targeting beats
> emailing everyone.
>
> Both emails work on average. I then built uplift models to find who
> responds to the women's email. On one test set, emailing the top 90
> percent looked 38 dollars per thousand customers better than emailing
> everyone. That rested on eight buyers, so I re-tested it with repeated
> cross-fitting on the full data, and it disappeared. No targeting policy
> beat emailing everyone.
>
> What the models had really found was one rule: people who only bought
> men's products barely respond to the women's email.
>
> The bigger finding was that the men's email works about equally for
> everyone, and sending it to all customers has the highest estimated
> profit. On spend that's suggestive, not proven, and it may be that
> particular creative, so I'd confirm it with a fresh test.

About 160 words, which is roughly 60 seconds at a normal speaking pace.

## 10 likely follow-up questions

No answers here on purpose.

1. Your Phase 3 result said the top 90% beat emailing everyone with a
   confidence interval that excluded zero. Why didn't you trust it, and what
   specifically told you to keep checking?
2. You valued policies with inverse propensity weighting on actual outcomes
   instead of summing the model's predicted uplift. Why does that matter,
   and what goes wrong if you use the predictions?
3. Explain a Qini curve to a product manager. What does it mean if the curve
   bends above the diagonal in the middle but not at the start?
4. The S-learner predicted exactly zero spend uplift for every customer.
   Why does that happen, and what would you change to fix it without
   peeking at the test set?
5. You trained the models on visits but the business cares about revenue.
   Defend that choice. When would it be the wrong one?
6. Your models were statistically tied, and you picked the causal forest
   anyway. How should a team choose between models when accuracy can't
   separate them?
7. A simple rule matched the model within $5–$9 per 1,000 customers. When
   is a machine-learning uplift model worth building and maintaining, and
   what would have had to be true of this data for it to pay off?
8. The men's email beat the women's email with p = 0.031 on spend. Would you
   ship "men's email to everyone" on that evidence? What would change your
   mind in either direction?
9. Suppose there had been no experiment and you only had logs of who
   marketing chose to email. Walk me through what you would do, and how
   confident you could be in the answer.
10. Design the follow-up experiment. What are the arms, the primary metric,
    the sample size, the duration, and the decision rule you would commit to
    before seeing data?
