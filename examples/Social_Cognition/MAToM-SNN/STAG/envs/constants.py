"""
Global constants for the gridworld env.
"""

# =============================================================================
# set the value of interface
# =============================================================================
FPS = 25
WinWidth = 340 #window width
WinHeight = 260 #window width
BoxSize = 20    #the size of one grid
GridWidth = 7   #the number of lattices are there in the x-axis
GridHeight = 7  #the number of lattices are there in the y-axis
XMargin = int((WinWidth - GridWidth * BoxSize)/2)
TopMargin = int((WinHeight - GridHeight * BoxSize))/2-5
# =============================================================================
# set color
# =============================================================================
White = (255, 255, 255)
Gray = (185, 185, 185)
Black = (0, 0, 0)
Red = (255, 0, 0)
Green = (0, 128, 0)
SpringGreen = (60, 179, 113)
DarkOrange = (255, 140, 0)
RoyalBlue = (65, 105, 225)
DarkVoilet = (148, 0, 211)
HotPink = (255, 105, 180)
BoardColor = White
BGColor = White
TextColor = White
Test = []
# =============================================================================
# set maps
# =============================================================================
# BlankBox = 1
# shadow = 0
# Wall = 5
# Obstacle = 5
# observer = 8
# button = 7
# obeservation_1 = 11
# obeservation_2 = 22
# obeservation_3 = 33
"""
'S' : starting point
'F' or '.': free space
'W' or 'x': wall
'H' or 'o': hole (terminates episode)
'G' : goal
"""
Start = 'S'
Free_space  = 'F'
Wall = 'W'
Danger = 'H'
Goal = 'G'
Shadow = 'Sh'


MAPs = {
    0: [
        "FFFFF",
        "FHFWF",
        "FFFFF",
        "WFFFF",
        "FFFGF"
    ],
    1: [
        "FFFFF",
        "FHWFF",
        "FFFFF",
        "WFGFF",
        "FFFFF"
    ],
    2: [
        "FFFF",
        "FWFW",
        "FFFW",
        "WFFG"
    ],

    3: [
        "FFFF",
        "FHFW",
        "FFFW",
        "GFFF"
    ],

    4: [
        "FFFF",
        "FHFW",
        "FFFW",
        "WGFF"
    ],
}
# 2: [
#     "SFFFFFFF",
#     "FFFFFFFF",
#     "FFFHFFFF",
#     "FFFFFWFF",
#     "FFFHFFFF",
#     "FWHFFFWF",
#     "FWFFHFWF",
#     "FFFWFFFG"
# ],
