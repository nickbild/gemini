import string
import random
import hashlib


num_seqs = 100
valid_chars = string.ascii_lowercase + " ."

for s in range(num_seqs):
    input_seq = ""
    for i in range(20):
        input_seq += random.choice(valid_chars)

    md5hex = hashlib.md5(input_seq.encode('utf-8')).hexdigest()

    print("{}\t{}".format(input_seq, md5hex))
