data_dir = "data/PANDA"
image_dir = f"{data_dir}/images"
label_dir = f"{data_dir}/labels"

class Config:
    # Full server endpoint pool.
    servers =[
        'tcp://192.168.137.10:5561',
        'tcp://192.168.137.20:5561',
        'tcp://192.168.137.10:5560',
        'tcp://192.168.137.20:5560',
        'tcp://192.168.137.251:5560',
        'tcp://192.168.137.251:5561',

    ]


    # Per-model server endpoint pools.
    servers_n =[
        'tcp://192.168.137.10:5561',
    ]
    servers_s = [
        'tcp://192.168.137.10:5561',
        'tcp://192.168.137.251:5561'
    ]
    servers_m = [
        'tcp://192.168.137.20:5561',
        'tcp://192.168.137.10:5560',
    ]
    servers_l = [

        'tcp://192.168.137.20:5560',
        'tcp://192.168.137.251:5560',
    ]
