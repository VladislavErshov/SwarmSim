import multiprocessing as mp



def worker_wrapper(arg):
    worker, kwargs = arg
    return worker(**kwargs)


def mp_kwargs_wrapper(worker, kwargs_list):
    pool=mp.Pool()
    arg = [(worker, kwargs) for kwargs in kwargs_list]
    result = pool.map(worker_wrapper, arg)
    return result


def sample_worker(a=1, b=2, c=3):
    return a*b*c


if __name__ == "__main__":    
    kwargs_list = [
        {'a' : 2, 'b' : 10},
        {'b' : 100},
        {'a' : 1, 'b' : 1, 'c' : 1},
        {'a': 100, 'b' : 100, 'c' : 100},
    ]
    res = mp_kwargs_wrapper(sample_worker, kwargs_list)
    print(res)