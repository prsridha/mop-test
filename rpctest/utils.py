

def execute_fn(model_checkpoint_path, input_fn_string, model_fn_string, train_fn_string):
    input_fn = dill.loads(base64.b64decode(input_fn_string))
    model_fn = dill.loads(base64.b64decode(model_fn_string))
    train_fn = dill.loads(base64.b64decode(train_fn_string))

    