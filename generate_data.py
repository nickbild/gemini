import random
import subprocess
from black_box import main


num_seqs_train = 500
num_seqs_test = 500
seq_length = 3
valid_chars = "0123456789"

unq = {}

file_train = open("data/train/input_pairs.txt", "w")
file_test = open("data/test/input_pairs.txt", "w")


def write_seqs_to_file(num_seqs, file):
    s = 0
    while s < num_seqs:
        input_seq = ""
        for i in range(seq_length):
            input_seq += random.choice(valid_chars)

        if input_seq not in unq:
            unq[input_seq] = 1
            input_seq = int(input_seq)

            result = main(input_seq)
            result = int(result)

            file.write("{}\t{}\n".format(input_seq, result))
            s += 1


if __name__ == "__main__":
    write_seqs_to_file(num_seqs_train, file_train)
    write_seqs_to_file(num_seqs_test, file_test)
