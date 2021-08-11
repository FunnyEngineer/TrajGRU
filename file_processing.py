import numpy as np
import struct

def read_bytes(data, height, width, pos):
    try:
        header_len, = struct.unpack(">l", data[0:4])
    except Exception:
        return -1 * np.ones((height, width))
    #if header_len != 512:
    #  raise ValueError(f"Header length should be 512, but {header_len} instead.")

    # header 1: start index: i * 2 + 2
    timeinfo = struct.unpack(">hhhhhh", data[4:16])
    # id 16 and 17: number of rows and number of columns in field
    n_row, = struct.unpack(">h", data[34:36])
    n_col, = struct.unpack(">h", data[36:38])

    # header 2: start index: (i * 2 - 31) * 2
    # id 34 and 36: top left corner's position
    TL_y, = struct.unpack(">f", data[74:78])
    TL_x, = struct.unpack(">f", data[82:86])


    #unit = struct.unpack(">cccccccc", data[358:358+8])

    # data length: 520 - 524
    data_len, = struct.unpack(">l", data[520:524])
    if data_len != n_row * n_col * 2:
        raise ValueError(f"Data length should be {n_row * n_col * 2}, but {data_len} instead.")
        # return -1 * np.ones((height, width))

    

    tx, ty = pos

    TL_c = int((tx - TL_x) // 1000)
    TL_r = int((TL_y - ty) // 1000)

    if TL_c > n_col or TL_r > n_row:
        raise ValueError(f"Nimrod's image top-left corner position is ({TL_x}, {TL_y}. Target top left corner is not covered by Nimrod image. Please check target area's top-left corner's coordinate.")
        
    if TL_c + width > n_col or TL_r + height > n_col:
        raise ValueError(f"Nimrod's image top-left corner position is ({TL_x}, {TL_y}. Nimrod's image contains {n_row} rows and {n_col} columns. Target area exceeds the edge of Nimrod image. Please check target area's height and weight.")

    start_idx = TL_r * n_col + TL_c
    delta_idx = n_col - width

    #print("****** Start extracting target rainfall data of {:4d}/{:2d}/{:2d} {:2d}:{:2d}:{:2d}. ******".format(*timeinfo))

    target_rain = []

    s = 524 + start_idx * 2
    e = s + width * 2
    for i in range(height):
        sub_rain = struct.unpack(">"+"h"*width, data[s:e])
        target_rain.append(sub_rain)
        s = e + delta_idx * 2
        e = s + width * 2
    
    target_rain = np.array(target_rain) / 32
    return target_rain