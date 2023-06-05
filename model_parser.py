import torchvision.transforms.functional as F
import torch.nn as nn
import re

parentheses_pattern = r"\((.*?)\)"
name_pattern = r".*\("


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.lstm = None

        for key, value in kwargs.items():
            setattr(self, key, value)


def search_pattern(string, pattern):
    match = re.search(pattern, string)
    if match:
        word = match.group(0)
        return word
    return None


def get_body(lines):
    string = '\n'.join(lines)
    opened_parentheses = -1
    start_pos = 0
    end_pos = -1

    result = string.strip()
    num_lines = len(lines)

    for i, s in enumerate(string):
        if s == '(':
            if opened_parentheses == -1:
                start_pos = i
                opened_parentheses += 2
            else:
                opened_parentheses += 1
        elif s == ')':
            opened_parentheses -= 1
        if opened_parentheses == 0:
            end_pos = i+1
            result = string[start_pos:end_pos].strip()
            break

    sum_chars = 0
    for j, line in enumerate(lines):
        sum_chars += len(line)
        if sum_chars >= end_pos:
            num_lines = j
            break
    return result, num_lines


def parse_model(model_repr):
    modules = {}
    lines = model_repr.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == "":
            i += 1
            continue
        elif '(' in line and ')' in line:
            module_type = search_pattern(line, parentheses_pattern)
            module_name = module_type.strip('()')
            start = line.split(':')[1]
            name = start.split('(')[0].strip()
            module_body, num_lines = get_body([start] + lines[i+1:])
            module = parse_module(module_body, name)
            num_lines = num_lines if num_lines > 0 else 1
            i += num_lines
            modules[module_name] = module
        else:
            i += 1
            continue
    return modules


def parse_module(module_body, name):
    layers = []
    for line in module_body.split('\n'):
        splitted = line.split(':')
        if len(splitted) == 2:
            func = splitted[1].strip()
            layers.append(parse_layer('nn.' + func))

    if name == 'Sequential':
        return nn.Sequential(*layers)
    elif name == 'Flatten':
        return parse_layer('nn.' + name + module_body)
    elif name == 'LSTM':
        return parse_layer('nn.' + name + module_body)
    else:
        raise ValueError(f"Invalid module name: {name}")


def parse_layer(layer_repr):
    return eval(layer_repr)


def forward(self, x):
    # x = F.normalize(x, mean=[0.5], std=[0.5])
    x = self.conv_layers(x)
    if self.lstm is not None:
        x = nn.AdaptiveAvgPool2d((1, 25))(x)  # Replace 'sequence_length' with the desired value

        x = x.squeeze().permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = hn[-1]
    x = self.flatten(x)
    x = self.fc_layers(x)
    return x


def create_model(model_string, forward_method=forward):
    modules = parse_model(model_string)

    model = CNN(**modules)
    CNN.forward = forward_method
    return model


if __name__ == '__main__':
    # Provide the model representation as a string
    model_repr = '''CNN(
  (conv_layers): Sequential(
    (0): Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Sigmoid()
    (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (4): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (6): Sigmoid()
    (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (8): Dropout(p=0.25, inplace=False)
  )
  (lstm): LSTM(128, 80, batch_first=True, bidirectional=True)
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (fc_layers): Sequential(
    (0): Linear(in_features=12800, out_features=2048, bias=True)
    (1): BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): Sigmoid()
    (3): Linear(in_features=2048, out_features=968, bias=True)
  )
)'''

    # Parse the model representation and create the PyTorch model
    model = create_model(model_repr)
    print(model)