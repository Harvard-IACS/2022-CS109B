def build_NN(name='NN', input_shape=(1,), hidden_dims=[2], hidden_act='relu', out_dim=1, out_act='linear'):
    model = Sequential(name=name)
    model.add(Input(shape=input_shape))
    for hidden_dim in hidden_dims:
        model.add(Dense(hidden_dim, activation=hidden_act))
    model.add(Dense(out_dim, activation=out_act))
    return model