from torchvision import datasets


def main():
    dataset = datasets.MNIST(
        './data',
        download=True)
    dataset[0][0].save('1.png')
    dataset[1][0].save('2.png')
    dataset[2][0].save('3.png')
    dataset[3][0].save('4.png')
    dataset[4][0].save('5.png')


if __name__ == '__main__':
    main()
