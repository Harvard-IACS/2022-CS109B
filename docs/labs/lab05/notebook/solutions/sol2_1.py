model = build_NN(name='penguins',
                 input_shape=(X_train.shape[1],),
                 hidden_dims=[8,16,3],
                 hidden_act='relu',
                 out_dim=y_train.shape[1],
                 out_act='softmax')
model.summary()