# RNN model
THis is a implementation of recurrent neural network which is a model that have better understanding of sequential data. If a user watched item have a lot to do with his/her next watched item. We suppose RNN model would give a better solution.

# Input & Output
Input : A dictionary of user_id : [a list of item_id] 

Output : Hit ratio of the model

# Result
| | RNN  |  NCF    |
|----------| ------------- |------|
|   hit@5    | 0.021       | 2.59       |
|   hit@10    | 0.052       | 4.62       |
|   hit@15    | 0.093       | 6.36       |
|   hit@20    | 0.116       | 7.91       |
|   hit@30    | 0.143       | 10.6       |

The columns represent RNN and NCF evaluation results on different top K hit ratio. However, we can observe that the NCF model seems to outperform RNN model in this datasets. The user bahavior seems to be influenced more by the content of movie itself than the movies he/she previously watched.
