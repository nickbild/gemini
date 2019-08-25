import sys
import math


def times_two_plus_seventeen(input):
    return (input * 2) + 17


def divide_by_four(input):
    return input / 4


def times_pi(input):
    return input * 3.14159


def if_under_200_subtract_six(input):
    if input < 200:
        input -= 6

    return input


def minus_seven(input):
    return input - 7


def main(input):
    output = times_two_plus_seventeen(input)
    output = math.floor(output)
    output -= 1
    output = minus_seven(output)
    output = divide_by_four(output)
    output = times_pi(output)
    output = math.ceil(output)
    output = if_under_200_subtract_six(output)
    output = math.floor(output)

    if output < 0:
        output = 0

    return output


if __name__ == "__main__":
    input = int(sys.argv[1])

    print(main(input))
