

def format_targets(target):
    input_target = target[:, :-1, :]
    output_target = target[:, 1:, :]
    return input_target, output_target