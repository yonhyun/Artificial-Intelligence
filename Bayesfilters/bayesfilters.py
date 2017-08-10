import numpy as np

world = np.zeros(shape=(8, 10))
world[1:5, 1] = 1
world[2:4, 3] = 1
world[4, 2:4] = 1
world[2, 4] = 1
world[2:7, 6] = 1
world[6, 2] = 1
world[6, 4:7] = 1
world[1:5, 8] = 1
world[6:8, 8] = 1


def neighbors(world, x, y):
    temp_list = []

    if (world[x, y] == 1):
        temp_list.append([x, y])
        return temp_list
    if (x == 0 and y == 0):
        temp_list.append([x + 1, y])
        temp_list.append([x, y + 1])
        temp_list.append([x, y])
        # print(temp_list)
        return temp_list
    elif (x == 0 and y == 9):
        temp_list.append([x + 1, y])
        temp_list.append([x, y - 1])
        temp_list.append([x, y])
        return temp_list
    elif (x == 7 and y == 0):
        temp_list.append([x, y + 1])
        temp_list.append([x - 1, y])
        temp_list.append([x, y])
        return temp_list
    elif (x == 7 and y == 9):
        temp_list.append([x - 1, y])
        temp_list.append([x, y - 1])
        temp_list.append([x, y])
        return temp_list
    elif (x == 0):
        temp_list.append([x + 1, y])
        temp_list.append([x, y - 1])
        temp_list.append([x, y + 1])
        temp_list.append([x, y])
        return temp_list
    elif (y == 0):
        temp_list.append([x + 1, y])
        temp_list.append([x - 1, y])
        temp_list.append([x, y + 1])
        temp_list.append([x, y])
        return temp_list
    elif (x == 7):
        temp_list.append([x, y + 1])
        temp_list.append([x, y - 1])
        temp_list.append([x - 1, y])
        temp_list.append([x, y])
        return temp_list
    elif (y == 9):
        temp_list.append([x - 1, y])
        temp_list.append([x + 1, y])
        temp_list.append([x, y - 1])
        temp_list.append([x, y])
        return temp_list
    else:
        temp_list.append([x - 1, y])
        temp_list.append([x, y - 1])
        temp_list.append([x + 1, y])
        temp_list.append([x, y + 1])
        temp_list.append([x, y])
        return temp_list


def ARandomWalk(world):
    tran_mat = np.zeros(shape=(80, 80))
    temp_tran = np.zeros(shape=80)
    total = []
    for index, value in np.ndenumerate(world):
        (x, y) = index

        possible = neighbors(world, x, y)

        num_of_possible = 0
        for [x1, y1] in possible:
            if world[x1, y1] == 0:
                num_of_possible = num_of_possible + 1

            else:
                if [x1, y1] in possible:
                    possible.remove([x1, y1])

        total.append(possible)
    for ind_y, value3 in enumerate(total):
        sample = 0
        if value3 != []:
            sample = len(value3)
            temp1 = np.zeros(shape=(8, 10))
            for [x, y] in value3:
                temp1[x, y] = round(1 / sample, 2)
        temp1 = np.reshape(temp1, 80)
        tran_mat[:, ind_y] = temp1

    return tran_mat


tran_mat = ARandomWalk(world)

left_corner = tran_mat[:, 0]


# p1=np.dot(tran_mat,left_corner)
# p2=np.dot(tran_mat,p1)
# keep multiply there is no change



def pss_f(p0, tran_mat):
    Ap = np.dot(tran_mat, p0).round(decimals=4)

    if np.array_equal(Ap, p0):

        p0 = np.reshape(p0, (8, 10))

        return p0

    else:

        return pss_f(Ap, tran_mat)


pss = pss_f(left_corner, tran_mat)


def obs(state, a):
    x = 0
    y = 0

    if state < 10:
        x = 0
        y = state
    else:
        x = state // 10
        y = state % 10
    naver = neighbors(a, x, y)
    wall = [1, 1, 1, 1]
    if a[x, y] == 1:
        return [3, 3, 3, 3]

    up = []
    left = []
    down = []
    right = []
    vec = []
    if naver != []:

        for [xx, yy] in naver:

            if y > yy:
                wall[0] = 0
                continue
            elif x > xx:
                wall[1] = 0
                continue
            elif y < yy:
                wall[2] = 0
                continue
            elif x < xx:
                wall[3] = 0
                continue
            else:
                vec.append(0)
                continue
    else:

        return wall

    return wall


# def pickwall(out):
#   temp=out
#  for index,p in np.ndenumerate(out):
#     (x,y)=index
#    if p>0.4:
#       temp[x,y]==0
# return temp


obs_output = obs(79, world)


def obslikelihood(a):
    overall = []
    int1 = 0
    for i in range(80):
        int1 = int1 + 1
        product = 1
        temp_1 = []
        temp_2 = obs(i, a)
        for k in temp_2:
            if k == 1:
                temp_1.append(0.8)
            elif k == 0:
                temp_1.append(0.2)
            elif k == 3:
                temp_1.append(0)
        for p in temp_1:
            product *= p
        overall.append(product)

    return overall


out = obslikelihood(world)
out = np.array(out)
obs_likeli = np.reshape(out, (8, 10))


def find_likeli(obs):
    temp1 = out
    candi_list = []

    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    candi = find_nearest(temp1, obs)
    for index, i in np.ndenumerate(obs_likeli):
        (x, y) = index
        if i == candi:
            candi_list.append([x, y])

    temp1 = np.reshape(temp1, (8, 10))
    return candi_list


find_likely = find_likeli(0.8 * 0.8 * 0.8 * 0.2)


def robotbayesfilter(prev, obs):
    new_p = np.dot(tran_mat, prev)
    return




