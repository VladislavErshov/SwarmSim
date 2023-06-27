import itertools



translation_table = str.maketrans('', '', ''.join(["'", ":", "{", "}", ","]))

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))