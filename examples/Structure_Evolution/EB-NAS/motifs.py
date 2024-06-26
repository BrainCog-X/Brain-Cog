from collections import namedtuple

import torch

Genotype = namedtuple('Genotype', 'normal normal_concat')

"""
Operation sets
"""

PRIMITIVES = [
    'skip_connect',
    # 'max_pool_3x3',
    # 'avg_pool_3x3',
    # 'def_conv_3x3',
    # 'def_conv_5x5',
    # 'sep_conv_3x3',
    # 'sep_conv_5x5',
    # 'dil_conv_3x3',
    # 'dil_conv_5x5',

    # 'max_pool_3x3_p',
    # 'avg_pool_3x3_p',
    'conv_3x3_p',
    'conv_5x5_p',
    # 'skip_connect_p',
    # 'sep_conv_3x3_p',
    # 'sep_conv_5x5_p',
    # 'dil_conv_3x3_p',
    # 'dil_conv_5x5_p',
    # 'def_conv_3x3_p',
    # 'def_conv_5x5_p',n

    # 'max_pool_3x3_n',
    # 'avg_pool_3x3_n',
    'conv_3x3_n',
    'conv_5x5_n',
    # 'skip_connect_n',
    # 'sep_conv_3x3_n',
    # 'sep_conv_5x5_n',
    # 'dil_conv_3x3_n',
    # 'dil_conv_5x5_n',
    # 'def_conv_3x3_n',
    # 'def_conv_5x5_n',

    # 'transformer',
]
m0=Genotype(
    normal=[
        ('skip', 0), ('skip', 1),('skip', 2),
    ],
    normal_concat=range(3, 4)
)
mm0=Genotype(
    normal=[
        ('skip', 0), ('skip', 1),('skip', 2),
    ],
    normal_concat=range(2, 3)
)
mm1=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),
        ('skip_connect', 0), ('conv_5x5_p', 2),
    ],
    normal_concat=range(2, 4)
)


mm2=Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_p', 1),
        ('skip_connect', 0), ('conv_5x5_n', 2),
        ('conv_5x5_p', 2), ('conv_3x3_n', 3),

    ],
    normal_concat=range(2, 5)
)

mm4=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),#2
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),#3
        ('conv_5x5_p', 2), ('conv_5x5_p', 3),#4
        ('skip_connect', 0), ('conv_3x3_p', 4),#5
        ('skip_connect', 0), ('conv_3x3_p', 4),#6
        ],
    normal_concat=range(2, 7)
)



mm3=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),#2
        ('skip_connect', 0), ('conv_5x5_n', 2),#3
        ('skip_connect', 0), ('conv_5x5_p', 3),#4

        ('skip_connect_back', 2),#3
        ('conv_3x3_p_back', 3),#4

    ],
    normal_concat=range(2, 5)
)


mm5=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),#2
        ('skip_connect', 0), ('conv_5x5_p', 2),#3

        ('skip_connect_back', 2),#3
    ],
    normal_concat=range(2, 4)
)



m1=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), ('conv_5x5_p', 2), #B3
        ('skip', 0), ('conv_5x5_p', 3), ('skip', 1), #C4
    ],
    normal_concat=range(3, 5)
)


m2=Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_p', 1),('conv_5x5_p', 2), #B3
        ('skip', 0), ('conv_5x5_n', 3), ('skip', 1),#C4
        ('conv_5x5_p', 3), ('conv_3x3_n', 4), ('skip', 1), #D5

    ],
    normal_concat=range(3, 6)
)

m4=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),('conv_5x5_p', 2), #3
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),('conv_5x5_p', 2), #4
        ('skip', 0), ('conv_5x5_p', 3), ('conv_5x5_p', 4), #5
        ('skip', 0), ('conv_3x3_p', 3),('conv_3x3_n', 5),#6
        ('skip', 0), ('conv_3x3_p', 4),('conv_3x3_n', 5),#7
        ],
    normal_concat=range(3, 8)
)



m3=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), ('conv_3x3_p', 2), #3
        ('skip', 0), ('conv_5x5_p', 3),('skip', 1), #4
        ('skip', 0), ('conv_5x5_p', 3), ('skip', 1), #5

        ('conv_3x3_n_back', 3),#4
        ('skip_back', 2),#5

    ],
    normal_concat=range(3, 6)
)


m5=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), ('conv_5x5_p', 2),#3
        ('skip', 0),('skip', 1), ('conv_5x5_n', 3), #4

        ('skip_connect_back', 3),#4
    ],
    normal_concat=range(3, 5)
)


t1=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), ('conv_5x5_p', 2),  ('conv_5x5_p', 3), #4
        ('skip', 0), ('conv_5x5_p', 4), ('skip', 1), ('skip', 2), #5
        ('skip', 0), ('conv_5x5_p', 5), ('skip', 1), ('skip', 2), #6
        ('skip', 0), ('conv_5x5_p', 5), ('skip', 1), ('skip', 2), #7
    ],
    normal_concat=range(4, 8)
)


t2=Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_p', 1),('conv_5x5_p', 2), ('conv_5x5_p', 3), #4
        ('skip', 0), ('conv_5x5_n', 4), ('skip', 1),('skip', 2),#5
        ('conv_5x5_p', 4), ('conv_3x3_n', 5), ('skip', 1),('skip', 2), #6

    ],
    normal_concat=range(4, 7)
)

t4=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),('conv_5x5_p', 2), ('conv_5x5_p', 3), #4
        ('conv_5x5_p', 0), ('skip', 1),('conv_5x5_n', 4), ('skip', 3), #5
        ('skip', 0), ('conv_5x5_p', 3), ('conv_5x5_n', 4), ('skip', 2),#6

        ],
    normal_concat=range(4, 7)
)



t3=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), ('conv_3x3_p', 2), ('conv_3x3_p', 3),#4
        ('skip', 0), ('skip', 2),('skip', 1), ('skip', 3),('conv_3x3_p', 4),#5
        ('skip', 0), ('conv_5x5_p', 4), ('skip', 1),('skip', 2), #6

        ('conv_3x3_n_back', 4),#5
        ('skip_back', 4),#6

    ],
    normal_concat=range(4, 7)
)


t5=Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_p', 1), ('conv_5x5_p', 2),('conv_5x5_p', 3),#4
        ('skip', 0),('skip', 1), ('skip', 2), ('conv_5x5_n', 4), #5
        ('skip', 0),('skip', 1), ('conv_5x5_n', 4),('conv_5x5_n', 5), #6
        ('skip', 0),('skip', 1), ('conv_5x5_n', 4),('conv_5x5_n', 5), #7

        ('conv_3x3_n_back', 4),#5
        ('skip_back', 4),#6
        ('skip_back', 5),#7

    ],
    normal_concat=range(4, 8)
)