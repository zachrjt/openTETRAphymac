import numpy as np


def generateHarmonics(Flo:float, Fclk:float, n:int):
    pass







fclk = 0
outputBW = 180E3
LOfreq = 164.175

half_db_bw_khz = (fclk/192)*1000 

# decimation factor can be 60 x M
# or 48 x M (where passband variation is 0.9dB and higher alias attenuation)