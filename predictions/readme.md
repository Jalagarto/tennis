# steps to get a good model for tabular data:

1. split data properly. see which steps are performed by gluon and which ones are not ... check documentation.  
2. Load and train model (use gpu to train faster and better).
3. Evaluate
4. Show leaderboard and perform evaluation on best model
5. [perform model distilation](https://auto.gluon.ai/stable/api/autogluon.task.html#autogluon.tabular.TabularPredictor.distill).
6. Evaluate again and see whether it improved
7. Perform xAI (anchors explanations!) [lime](https://github.com/marcotcr/lime)
