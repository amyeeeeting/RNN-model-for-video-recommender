# RNN model
THis is a implementation of recurrent neural network which is a model that have better understanding of sequential data. If a user watched item have a lot to do with his/her next watched item. We suppose RNN model would give a better solution.

# Input & Output
Input : A dictionary of user_id : [a list of item_id] 
Output : Hit ratio of the model

# Result
| | RNN  |  NCF    | Validation Loss   |Test Loss (Scale Inverted) |
|----------| ------------- |------|-------| -----|
|   hit@5    | 2       | ReLU       |    0.00029     | 114308 |
|   hit@10    | 2       | Leaky ReLU       |    0.00029     | 115525 |
|   hit@15    | 3       | ReLU       |    0.00029     | 201718 |
|   hit@20    | 3       | Leaky ReLU       |    0.00028     | 108700 |
