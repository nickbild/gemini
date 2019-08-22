import sys
import math


def square_plus_seventeen(input):
    return (input ** 2) + 17


def divide_by_four(input):
    return input / 4


def times_pi(input):
    return input * 3.14159


def minus_seven(input):
    return input - 7


def filter(input):
    if input < 100:
        return input / 2

    return input


def remove_20s(input, output):
    if input >= 20 and input <= 29:
        return 0

    return output


def remove_110s(input, output):
    if input >= 100 and input <= 109:
        return 0

    return output


def main(input):
    output = square_plus_seventeen(input)
    output = filter(output)
    output = remove_20s(input, output)
    output = remove_110s(input, output)
    output = math.floor(output)
    output -= 1
    output = minus_seven(output)
    output = divide_by_four(output)
    output = times_pi(output)
    output = math.ceil(output)

    if output < 0:
        output = 0

    return output


if __name__ == "__main__":
    input = int(sys.argv[1])

    print(main(input))
