import torch


def main():
    x = torch.jit.load("cpp_tensor.pt", map_location='cpu')
    x = list(x.parameters())[0]
    print(x)


if __name__ == '__main__':
    main()
