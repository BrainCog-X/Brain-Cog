from collections import namedtuple

import torch

Genotype = namedtuple('Genotype', 'normal normal_concat')

"""
Operation sets
"""

PRIMITIVES = [
    'conv_3x3_p',
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
    # 'conv_3x3_p_p',
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
    # 'conv_3x3_p_n',
    # 'sep_conv_3x3_n',
    # 'sep_conv_5x5_n',
    # 'dil_conv_3x3_n',
    # 'dil_conv_5x5_n',
    # 'def_conv_3x3_n',
    # 'def_conv_5x5_n',

    # 'transformer',
]
"""====== SnnMlp Archirtecture By Other Methods"""

mlp1 = Genotype(
    normal=[
        ('mlp', 0), ('conv_3x3_p', 1),  # 2
        ('mlp', 1), ('mlp', 0),  # 3
        ('conv_3x3_p', 2), ('mlp', 3),  # 4
        ('mlp_back', 2),
        ('conv_3x3_p_back', 2)
    ],
    normal_concat=range(2, 5)
)

mlp2 = Genotype(
    normal=[
        ('mlp', 0), ('conv_3x3_p', 1),
        ('conv_3x3_p', 2), ('mlp_p', 1),
        # ('mlp_n', 1), ('conv_3x3_p', 2),
        ('mlp_back', 2)
    ],
    normal_concat=range(2, 4)
)


"""====== SNN Archirtecture By Other Methods"""

dvsc10_new_skip22 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),  # 2
        ('conv_5x5_p', 1), ('conv_3x3_p', 2),  # 3
        ('conv_3x3_p', 0), ('conv_3x3_p', 3),  # 4
        ('conv_3x3_n_back', 2), ('conv_3x3_p_back', 3)  # 3, 4
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip22 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_n', 1), ('conv_3x3_p', 2),
        ('conv_5x5_n', 0), ('conv_3x3_p', 3),
        ('conv_3x3_n_back', 0), ('conv_3x3_p_back', 1)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip21 = Genotype(
    normal=[
        ('conv_3x3_n', 0), ('conv_5x5_p', 1),  # 2
        ('conv_3x3_p', 1), ('conv_5x5_p', 2),  # 3
        ('conv_5x5_n', 2), ('conv_3x3_p', 1),  # 4
        # ('conv_3x3_p_back', 2), ('conv_5x5_p_back', 2)
    ],
    normal_concat=range(2, 5)
)


dvsc10_new_skip20 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n', 2), ('conv_5x5_p', 0),
        ('conv_3x3_p', 2), ('conv_3x3_n', 3),
        ('conv_3x3_p_back', 2),
        ('conv_5x5_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip19 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_3x3_p', 1),
        ('conv_5x5_n', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_3x3_p_back', 2),
        ('conv_5x5_p_back', 2)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip18 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_5x5_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip17 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_5x5_n', 0),
        ('conv_5x5_n', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('avg_pool_3x3_p', 3),
        ('avg_pool_3x3_p_back', 2), ('conv_3x3_p_back', 2)
    ],
    normal_concat=range(2, 5)
)
dvsc10_new_skip16 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n', 2), ('avg_pool_3x3_p', 0),
        ('conv_3x3_p', 2), ('avg_pool_3x3_n', 3),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip15 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_p', 1),
        ('conv_5x5_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip14 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 1), ('conv_3x3_p', 2),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip13 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_n', 1), ('conv_3x3_p', 2),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip12 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_3x3_p', 1),
        ('conv_5x5_n', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)
    ],
    normal_concat=range(2, 5)
)
# dvsc10_new_skip12 = Genotype(
#     normal=[
#         ('conv_3x3_p', 0), ('conv_3x3_n', 1),
#         ('conv_3x3_p', 1), ('conv_5x5_n', 2),
#         ('conv_3x3_n', 3), ('conv_3x3_p', 0),
#         ('conv_5x5_p_back', 2), ('conv_3x3_p_back', 3)
#     ],
#     normal_concat=range(2, 5)
# )

dvsc10_new_skip11 = Genotype(normal=[
    ('conv_3x3_n', 0), ('conv_5x5_n', 1),
    ('conv_5x5_p', 0), ('conv_3x3_n', 2),
    ('conv_3x3_p', 2), ('conv_5x5_n', 0),
    ('conv_3x3_n_back', 2),
    ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip10 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip9 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_5x5_n', 0),
        ('conv_5x5_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_3x3_p_back', 2),
        ('conv_5x5_n_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip8 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n', 2), ('conv_5x5_p', 0),
        ('conv_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip7 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_n', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip6 = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_5x5_n', 1),
        ('conv_5x5_p', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_n_back', 2), ('conv_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip5 = Genotype(
    normal=[
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_n', 2), ('conv_3x3_n', 0),
        ('conv_3x3_p', 2), ('conv_3x3_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip4 = Genotype(
    normal=[
        ('conv_5x5_n', 1), ('conv_5x5_p', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 1),
        ('conv_3x3_p', 2), ('conv_5x5_n', 0),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip3 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_n', 2), ('conv_3x3_n', 0),
        ('conv_3x3_p', 2), ('conv_5x5_p', 3),
        ('conv_5x5_n_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_skip2 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('avg_pool_3x3_n', 0),
        ('avg_pool_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip1 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_n', 2), ('conv_3x3_p', 1),
        ('conv_5x5_p', 1), ('conv_3x3_p', 2),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_skip = Genotype(
    normal=[
        ('conv_3x3_n', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('conv_3x3_n', 0),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_base0 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_n', 3),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new_base1 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_5x5_n', 0),
        ('conv_5x5_p', 1), ('conv_3x3_p', 0),
        ('conv_5x5_n', 1), ('conv_3x3_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('conv_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5)
)

dvsc10_new_base2 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_3x3_p', 1),
        ('conv_5x5_n', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n', 3), ('conv_5x5_n', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new_base3 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_p', 1), ('conv_3x3_n', 0),
        ('conv_5x5_p', 1), ('conv_3x3_n', 0),
        ('conv_3x3_p_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_grad2 = Genotype(
    normal=[
        ('avg_pool_3x3_n', 1), ('conv_5x5_p', 0),
        ('conv_5x5_n', 1), ('conv_5x5_n', 0),
        ('conv_3x3_p', 3), ('conv_5x5_n', 1),
        ('conv_5x5_p_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_grad1 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_n', 2), ('avg_pool_3x3_n', 1),
        ('avg_pool_3x3_p', 2), ('conv_5x5_n', 1),
        ('conv_5x5_p_back', 2),
        ('conv_3x3_p_back', 3)],
    normal_concat=range(2, 5))

dvsg_new2 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5))

dvsg_new1 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 1),  ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('conv_5x5_n_back', 3)],
    normal_concat=range(2, 5))

dvscal_new1 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_n', 1),
        ('conv_5x5_n', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_p', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new8 = Genotype(
    normal=[('conv_5x5_p', 0), ('conv_5x5_p', 1),
            ('conv_3x3_p', 0), ('conv_5x5_n', 1),
            ('conv_5x5_p', 0), ('conv_5x5_n', 1),
            ('avg_pool_3x3_n_back', 2),
            ('avg_pool_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)

dvsc10_new7 = Genotype(
    normal=[
        ('conv_5x5_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_p', 0), ('conv_5x5_n', 1),
        ('conv_5x5_p', 0), ('conv_5x5_n', 1),
        ('conv_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5))

dvsc10_new6 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5))

dvsc10_new5 = Genotype(
    normal=[
        ('conv_5x5_p', 1), ('conv_3x3_p', 0),
        ('conv_3x3_p', 0), ('conv_5x5_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5))

dvsc10_new4 = Genotype(
    normal=[
        ('conv_3x3_n', 1), ('conv_3x3_p', 0),
        ('conv_5x5_p', 1), ('conv_5x5_p', 0),
        ('conv_5x5_p', 1), ('conv_5x5_p', 0),
        ('avg_pool_3x3_p_back', 2),
        ('avg_pool_3x3_n_back', 2)],
    normal_concat=range(2, 5),
)

dvsc10_new3 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_n', 1), ('conv_3x3_n', 0),
        ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_p', 2)],
    normal_concat=range(2, 5),
)

dvsc10_new2 = Genotype(normal=[
    ('conv_3x3_p', 0), ('conv_3x3_n', 1),
    ('conv_3x3_n', 1), ('avg_pool_3x3_p', 0),
    ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
    ('avg_pool_3x3_n_back', 2),
    ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5),
)

dvsc10_new1 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_p', 0), ('conv_3x3_n', 1),
        ('conv_3x3_p', 0), ('conv_3x3_p', 1),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_n_back', 2)],
    normal_concat=range(2, 5)
)

dvsc10_new0 = Genotype(
    normal=[
        ('conv_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('avg_pool_3x3_p', 2), ('conv_3x3_n', 1),
        ('conv_3x3_p', 0), ('conv_3x3_p', 3),
        ('conv_3x3_p_back', 2),
        ('conv_3x3_n_back', 3)],
    normal_concat=range(2, 5)
)
cifar_new_skip1 = Genotype(
    normal=[
        ('conv_5x5_n', 0), ('conv_5x5_p', 1),
        ('avg_pool_3x3_p', 0), ('avg_pool_3x3_n', 2),
        ('avg_pool_3x3_p', 2), ('conv_5x5_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('avg_pool_3x3_p_back', 3)
    ],
    normal_concat=range(2, 5))

cifar_new1 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_p', 0),
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),
        ('avg_pool_3x3_p', 2), ('conv_3x3_p', 0),
        ('avg_pool_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5)
)

cifar_new2 = Genotype(
    normal=[
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 0), ('avg_pool_3x3_p', 1),
        ('conv_3x3_p', 2), ('conv_3x3_n', 0),
        ('conv_3x3_n_back', 2),
        ('conv_3x3_p_back', 2)],
    normal_concat=range(2, 5),
)

cifar_new0 = Genotype(
    normal=[
        ('avg_pool_3x3_p', 1), ('avg_pool_3x3_n', 0),  # 2, 3
        ('conv_3x3_n', 0), ('avg_pool_3x3_p', 1),  # 4, 5
        ('conv_3x3_p', 2), ('conv_3x3_n', 3),  # 6 , 7
        ('avg_pool_3x3_n_back', 2),
        ('conv_3x3_p_back', 1)],
    normal_concat=range(2, 5)
)
