import string
import random
import hashlib


num_seqs = 50
seq_length = 10
valid_chars = "0123456789" # string.ascii_lowercase + " ."


for s in range(num_seqs):
    input_seq = ""
    for i in range(seq_length):
        input_seq += random.choice(valid_chars)

    md5hex = hashlib.md5(input_seq.encode('utf-8')).hexdigest()

    print("{}\t{}".format(input_seq, md5hex))
