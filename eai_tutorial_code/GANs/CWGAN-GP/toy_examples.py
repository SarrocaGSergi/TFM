from utils import *

dictionary = {0: "BACKPACKS",
                      1: "BAG ACCESSORIES",
                      2: "BELTS & SUSPENDERS",
                      3: "BLANKETS",
                      4: "BOAT SHOES & MOCCASINS",
                      5: "BOOTS",
                      6: "BRIEFCASES",
                      7: "CLUTCHES & POUCHES",
                      8: "DRESSES",
                      9: "DUFFLE & TOP HANDLE BAGS",
                      10: "DUFFLE BAGS",
                      11: "ESPADRILLES",
                      12: "EYEWEAR",
                      13: "FINE JEWELRY",
                      14: "FLATS",
                      15: "GLOVES",
                      16: "HATS",
                      17: "HEELS",
                      18: "JACKETS & COATS",
                      19: "JEANS",
                      20: "JEWELRY",
                      21: "JUMPSUITS",
                      22: "KEYCHAINS",
                      23: "LACE UPS",
                      24: "LINGERIE",
                      25: "LOAFERS",
                      26: "MESSENGER BAGS",
                      27: "MESSENGER BAGS & SATCHELS",
                      28: "MONKSTRAPS",
                      29: "PANTS",
                      30: "POCKET SQUARES & TIE BARS",
                      31: "POUCHES & DOCUMENT HOLDERS",
                      32: "SANDALS",
                      33: "SCARVES",
                      34: "SHIRTS",
                      35: "SHORTS",
                      36: "SHOULDER BAGS",
                      37: "SKIRTS",
                      38: "SNEAKERS",
                      39: "SOCKS",
                      40: "SUITS & BLAZERS",
                      41: "SWEATERS",
                      42: "SWIMWEAR",
                      43: "TIES",
                      44: "TOPS",
                      45: "TOTE BAGS",
                      46: "TRAVEL BAGS",
                      47: "UNDERWEAR & LOUNGEWEAR"}

print(dictionary)

batch = ['TOPS', 'TIES', 'SKIRTS']

z = []
for i in batch:
    for x in dictionary:
        if i == dictionary[x]:
            z.append(x)

print(z)

