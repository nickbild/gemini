import string
import random
import hashlib


num_seqs_train = 9984
num_seqs_test = 16
seq_length = 4
valid_chars = "0123456789" # string.ascii_lowercase + " ."

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
            md5hex = hashlib.md5(input_seq.encode('utf-8')).hexdigest()
            file.write("{}\t{}\n".format(input_seq, md5hex))
            s += 1


if __name__ == "__main__":
    write_seqs_to_file(num_seqs_train, file_train)
    write_seqs_to_file(num_seqs_test, file_test)
