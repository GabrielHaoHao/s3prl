from logging.config import valid_ident
import random


def select_snr_t():
    data = {10:50, 11:80, 12:90, 13:100, 14:100, 15:100, 16:80, 17:40, 18:20, 19:10, 20:10}

    value_sum = sum(data.values())
    t = random.randint(0, value_sum-1)
    for key,value in data.items():
        t -= value
        if t < 0:
            pick_value = key
            return pick_value

def select_snr_s():
    data = {5:10, 6:10, 7:10, 8:10, 9:10, 10:10, 11:10, 12:10, 13:10, 14:10, 15:10}
    value_list = []
    for key, value in data.items():
        value_list += value*[key]
    return value_list

if __name__ == "__main__":
    print(select_snr_s())