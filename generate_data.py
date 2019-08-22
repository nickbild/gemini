import random
import subprocess
from black_box import main


num_seqs_train = 250
num_seqs_test = 750
seq_length = 3
min_in = 0
max_in = 999
min_out = 0
max_out = 783835
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
            # Normalize output.
            input_seq = int(input_seq)
            input_seq_norm = (input_seq - min_in) / (max_in - min_in)

            unq[input_seq] = 1
            result = main(input_seq)

            # Normalize result.
            result = int(result)
            result_norm = (result - min_out) / (max_out - min_out)

            file.write("{}\t{}\n".format(input_seq_norm, result_norm))
            s += 1


if __name__ == "__main__":
    write_seqs_to_file(num_seqs_train, file_train)
    write_seqs_to_file(num_seqs_test, file_test)
