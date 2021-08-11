import zipfile
import tarfile
import gzip
import numpy as np
import pandas as pd
import struct
import re
import os
from tqdm import tqdm

args = {"start_time": "2020-04-01",
      "end_time": "2020-04-02",
      "target_top_left_pos": (149000, 544000),
      "side_len": 512,
      "np_file_name": "test.npy"}

def get_nimrod_by_date(args):
  """The Nimrod data covers time from 2014/01/01 to 2020/12/31."""

  # read arguments
  start_time = args["start_time"]
  end_time = args["end_time"]
  side_len = args["side_len"]
  pos = args["target_top_left_pos"]
  np_file_name = args["np_file_name"]

  # create file name of target time
  time_pattern = re.compile(r'\d+')
  s_year, s_month, s_day = [x if len(x) > 1 else "0" + x for x in time_pattern.findall(start_time)]
  e_year, e_month, e_day = [x if len(x) > 1 else "0" + x for x in time_pattern.findall(end_time)]

  zipfile_name = "Nimrod_2014-2020.zip"

  start_tarfile_name = f"Nimrod_2014-2020/{s_year}/metoffice-c-band-rain-radar_uk_{s_year}{s_month}{s_day}_1km-composite.dat.gz.tar"
  end_tarfile_name = f"Nimrod_2014-2020/{e_year}/metoffice-c-band-rain-radar_uk_{e_year}{e_month}{e_day}_1km-composite.dat.gz.tar"
  
  name_5min_list = [f"000{h*100+m*5}"[-4:] for h in range(24) for m in range(12)]

  # Open Nimrod_2014-2020.zip
  with zipfile.ZipFile(zipfile_name, "r") as z:
    raw_list = z.namelist()

    # Remove folders' names from list
    for y in range(2014, 2021):
      raw_list.remove(f"Nimrod_2014-2020/{y}/")

    # Get the indices of tarfiles of target time range 
    start_tarfile_idx = raw_list.index(start_tarfile_name)
    end_tarfile_idx = raw_list.index(end_tarfile_name)
    target_tarfile_list = raw_list[start_tarfile_idx:end_tarfile_idx+1]
    
    # Extract target tarfile
    for file in target_tarfile_list:
      z.extract(file)

    name_pattern = re.compile(r'\d{12}')

    target_rain = []

    
    for tarfile_name in target_tarfile_list:
      with tarfile.open(tarfile_name) as t:
        gz_list = t.getnames()
        gz_list.sort()

        """if len(gz_list) < 288:
          temp_gz_name = gz_list[0]
          idx = temp_gz_name.index(name_pattern.findall(temp_gz_name)[0])
          missing_list = [temp_gz_name[0:idx+8] + i + temp_gz_name[idx+12:] for i in name_5min_list]
          for i in gz_list:
            missing_list.remove(i)
          print(f"The missing files: {missing_list}")"""

        print(f"Processing {tarfile_name}")
        for gz in tqdm(t.getnames()):
          t.extract(gz, path="./target")
          
          with gzip.open("./target/"+gz, mode="rb") as g:
            data = g.read()
            target_rain.append(read_bytes(data, side_len, pos))

    target_rain = np.array(target_rain)
    print(f"The shape of target rain array is {target_rain.shape}")

    #np.save(np_file_name, target_rain)
  return target_rain

def read_bytes(data, side_len, pos):
  header_len, = struct.unpack(">l", data[0:4])
  if header_len != 512:
    raise(f"Header length should be 512, but {header_len} instead.")

  # header 1: start index: i * 2 + 2
  timeinfo = struct.unpack(">hhhhhh", data[4:16])
  # id 16 and 17: number of rows and number of columns in field
  n_row, = struct.unpack(">h", data[34:36])
  n_col, = struct.unpack(">h", data[36:38])

  # header 2: start index: (i * 2 - 31) * 2
  # id 34 and 36: top left corner's position
  TL_y, = struct.unpack(">f", data[74:78])
  TL_x, = struct.unpack(">f", data[82:86])

  # data length: 520 - 524
  data_len, = struct.unpack(">l", data[520:524])
  if data_len != n_row * n_col * 2:
    raise(f"Data length should be {n_row * n_col * 2}, but {data_len} instead.")

  tx, ty = pos

  TL_c = int((tx - TL_x) // 1000)
  TL_r = int((TL_y - ty) // 1000)

  if TL_c > n_col or TL_r > n_row:
    raise(f"Nimrod's image top-left corner position is ({TL_x}, {TL_y}. Target top left corner is not covered by Nimrod image. Please check target area's top-left corner's coordinate.")
    
  if TL_c + side_len > n_col or TL_r + side_len > n_col:
    raise(f"Nimrod's image top-left corner position is ({TL_x}, {TL_y}. Nimrod's image contains {n_row} rows and {n_col} columns. Target area exceeds the edge of Nimrod image. Please check target area's side length.")

  start_idx = TL_r * n_col + TL_c
  delta_idx = n_col - side_len

  #print("****** Start extracting target rainfall data of {:4d}/{:2d}/{:2d} {:2d}:{:2d}:{:2d}. ******".format(*timeinfo))

  target_rain = []

  s = 524 + start_idx * 2
  e = s + side_len * 2
  for i in range(side_len):
    sub_rain = struct.unpack(">"+"h"*side_len, data[s:e])
    target_rain.append(sub_rain)
    s = e + delta_idx * 2
    e = s + side_len * 2
  
  target_rain = np.array(target_rain)
  return target_rain

target_rain_array = get_nimrod_by_date(args)
