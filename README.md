# school-projects

Here is a brief introduction to all the projects that I was involvoed in when I studied in ENSAE ParisTech

1. Goldman Sachs annual report and stock price analysis
    
  context:
      We are interested in the short term effect a financial statement published by a company has on the companies stock    price. We use a multiple linear regression model similar to the model proposed by Patell [1976].
Our data consists of daily stock price changes of the banking institution Goldman Sachs and the KBW index, an index of the thirty biggest banks in the United States over a given period of time. Note that Goldman Sachs is not part of it. Furthermore, we add a dummy variable that indicates for any date during the time interval wether or not GS published a certain type of financial statement on that date. We end up with a data frame with three variables in n dates. We then regress the daily stock price change of Goldman and Sachs on the change of the KBW index and on the dummy. As opposed to Patell [1976] we use daily changes instead of weekly changes, we do not regress on a constant (doing so would lead to coefficient near zero) and we ignore dividend payouts.

main skills:  Web scrapping, linear regression, data visulization.

2.
