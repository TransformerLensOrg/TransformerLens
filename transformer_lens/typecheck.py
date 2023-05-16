import torch
import typeguard

def typecheck_fail_callback(err, memo):
    """Callback for typeguard to print out tensor shapes and dtypes on type errors"""
    tensor_arg_names = tuple(key for key in memo.locals.keys() if isinstance(memo.locals[key], torch.Tensor))
    if tensor_arg_names:
        max_arg_name_len = max(len(arg_name) for arg_name in tensor_arg_names)
        arg_shape_strs = tuple(str(tuple(memo.locals[arg_name].shape)) for arg_name in tensor_arg_names)
        max_arg_shape_len = max(len(arg_shape_str) for arg_shape_str in arg_shape_strs)
        tensor_arg_shapes = (f"{arg_name:<{max_arg_name_len + 1}} shape={arg_shape_str:<{max_arg_shape_len + 1}} dtype={memo.locals[arg_name].dtype}" for arg_name, arg_shape_str in zip(tensor_arg_names, arg_shape_strs))
        err_msg = str(err)
        err_msg += "\ntensor argument info:\n\t"
        err_msg += "\n\t".join(tensor_arg_shapes)
        raise typeguard.TypeCheckError(err_msg)
    raise err
