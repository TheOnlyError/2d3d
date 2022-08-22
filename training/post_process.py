import time
from functools import partial
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import cv2
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.path import Path
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform, sqeuclidean

gap_classes = {
    'r3d': [1],
    'cubicasa5k': [1],
    'cubicasa5k_test': [1],
    'multi': [1, 2]
}

cluster_classes = {
    'r3d': [2],
    'cubicasa5k': [3, 4, 5],
    'cubicasa5k_test': [4],
    'multi': [4, 5, 6, 7]
    # 'multi': [4, 6]
}

dataset_classes = {
    'r3d': list(range(3)),
    'cubicasa5k': list(range(6)),
    'cubicasa5k_test': list(range(6)),
    'multi': list(range(8)),
    'combi': list(range(8))
}

dataset_classes_map = {
    'r3d': {
        'walls': 1,
        'openings': 1,
    },
    'cubicasa5k': {
        'walls': 1,
        'railings': 2,
        'doors': 2,
        'windows': 2,
        'stairs': 2,
    },
    'multi': {
        'walls': 1,
        'glass_walls': 2,
        'railings': 3,
        'doors': 4,
        'sliding_doors': 5,
        'windows': 6,
        'stairs': 7,
    },
}

# type = 'r3d'
# type = 'cubicasa5k_test'
# type = 'cubicasa5k'
type = 'multi'


def plot_polygons(bg, vertices, polygons):
    bg = np.zeros(bg)
    plt.figure(dpi=300)
    plt.imshow(bg)
    for e in polygons:
        endpoints = np.stack(np.array(e)[:, 0], axis=0)
        endpoints = rotational_sort(endpoints)
        c = ['r', 'g', 'b', 'orange']
        for i, endpoint in enumerate(endpoints):
            if i > 3:
                continue
            plt.scatter([endpoint[0]], [endpoint[1]], c=c[i], s=(i + 1) / 2, alpha=0.4)
        plt.gca().add_patch(Polygon(endpoints, closed=True, facecolor='g', alpha=0.6))
    plt.show()


def fill_break_line(result):
    min_size = 5
    num_kernels = 2
    kernels = []
    for i in range(min_size, min_size + num_kernels * 2, 2):
        kernel1 = np.zeros((i, i), dtype=np.uint8)
        kernel2 = np.zeros((i, i), dtype=np.uint8)
        half = int(i / 2)
        kernel1[half, 0] = 1
        kernel1[half, -1] = 1
        kernel2[half, 0:half] = 1
        kernel2[half, half + 1:] = 1
        kernels.append(kernel1)
        kernels.append(kernel2)
    # kernels.append(np.array([
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0],
    #     [1, 1, 0, 1, 1],
    #     [0, 0, 1, 0, 0],
    #     [0, 0, 1, 0, 0]
    # ], np.uint8))
    # kernels.append(np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [1, 1, 0, 1, 1],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0]
    # ], np.uint8))
    # kernels.append(np.array([
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [1, 0, 0, 0, 1],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0]
    # ], np.uint8))
    # kernels.append(np.array([
    #     [1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 1]
    # ], np.uint8))
    # kernels.append(np.array([
    #     [1, 0, 0, 0, 0],
    #     [0, 1, 0, 0, 0],
    #     [1, 1, 0, 1, 1],
    #     [0, 0, 0, 1, 0],
    #     [0, 0, 0, 0, 1]
    # ], np.uint8))
    # kernels.append(np.array([
    #     [1, 0, 1, 0, 0],
    #     [0, 1, 1, 0, 0],
    #     [0, 0, 0, 0, 0],
    #     [0, 0, 1, 1, 0],
    #     [0, 0, 1, 0, 1]
    # ], np.uint8))

    for c in gap_classes[type]:
        img = (result == c).astype(np.uint8)
        for kernel_h in kernels:
            kernel_v = np.transpose(kernel_h)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_h)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_v)
        result[img == 1] = c
    return result


def rotational_sort(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(angles)
    return list_of_xy_coords[indices]


def angle_between_three_points(a, b, c):
    BA = a - b
    BC = c - b

    cosine_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    # angle = np.arccos(cosine_angle)
    return abs(cosine_angle)


def vertex_equal(a, b):
    return points_equal(a[0], b[0]) and a[1] == b[1]


def points_equal(a, b):
    return a[0] == b[0] and a[1] == b[1]


def merged_set(s, merged):
    result = set()
    for p in s:
        result.add(p)
        result = result.union(merged[p])
    return result


def segment_point(a, b, c):
    x1, y1 = a
    x2, y2 = c
    x3, y3 = b
    dy, dx = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = (dy * (y3 - y1) + dx * (x3 - x1)) / det
    return np.array([x1 + a * dx, y1 + a * dy], dtype=np.float32)


def process_tile(initial_len, img, show, polygons, new_stats, stat_ind):
    i, stat = stat_ind
    # stat = [ [x,y,w,h], [tile] ]
    x, y, w, h, _ = stat[0]
    tile = stat[1]
    if i < initial_len and w < 4 and h < 4:
        # print('Filter out')
        return
    if w < 2 or h < 2:
        # print('Filter out small')
        return

    bb = np.zeros(img.shape, np.uint8)
    bb[y: y + h, x: x + w] = img[y: y + h, x: x + w]
    if tile is not None:
        bb_tile = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(bb_tile, [tile], color=1)
        bb_tile *= img
        # bb = bb_tile
        bb = np.logical_and(bb, bb_tile) * img

    # Find min rotated rect
    inds = np.argwhere(bb > 0)
    if inds.shape[0] <= 2:
        return
    rect = cv2.minAreaRect(np.array(inds))
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    box[:, [1, 0]] = box[:, [0, 1]]

    mask = np.zeros(bb.shape, np.uint8)
    cv2.fillPoly(mask, [box], color=1)
    bb *= mask

    if show:
        plt.figure(dpi=200)
        plt.imshow(bb)
        # plt.gca().add_patch(
        #     Polygon([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], closed=True, facecolor='r', alpha=0.1))
        plt.gca().add_patch(Polygon(box, closed=True, facecolor='g', alpha=0.6))
        plt.axis('off')
        plt.savefig('results/bb_' + str(i), bbox_inches='tight', pad_inches=0)
        plt.show()

    # Break up squares
    d1 = np.linalg.norm(box[0] - box[1])
    d2 = np.linalg.norm(box[1] - box[2])
    d_min = min(d1, d2) + 1
    d_max = max(d1, d2) + 1

    # if w < 4 and h < 4:
    #     print('Filter out')
    #     # bd_ind[y: y + h, x: x + w] = 0
    #     continue

    # e = 0.35  # constant to determine when to refine
    e = 0.75  # constant to determine when to refine
    # e = 0.8  # constant to determine when to refine
    e = min(e, 4 / d_min)
    ratio = 1 - np.count_nonzero(bb) / (d_min * d_max)
    is_square = ratio >= e
    small_dim = 8
    small_square = d_min <= small_dim and d_max <= small_dim

    # print(ratio, e)
    if ratio >= 0.96:
        return
    if is_square and not small_square:
        if show:
            print(min(h, w) / max(h, w), ratio, h, w, box[0])
        points = rotational_sort(box)
        d1 = np.linalg.norm(points[0] - points[1])
        d2 = np.linalg.norm(points[1] - points[2])
        if d1 > d2:
            tiles = [
                [points[0], (points[0] + points[1]) / 2, (points[2] + points[3]) / 2, points[3]],
                [(points[0] + points[1]) / 2, (points[2] + points[3]) / 2, points[2], points[1]],
            ]
        else:
            tiles = [
                [points[0], points[1], (points[1] + points[2]) / 2, (points[0] + points[3]) / 2],
                [(points[1] + points[2]) / 2, (points[0] + points[3]) / 2, points[3], points[2]],
            ]

        for tile in tiles:
            # tile = np.array(tile).round().astype(np.int32)
            tile = np.array(tile, np.int32)
            new_tile = np.zeros(img.shape, np.uint8)
            cv2.fillPoly(new_tile, [tile], color=1)
            new_tile *= img
            _, _, square_stats, _ = cv2.connectedComponentsWithStats(new_tile)

            # if show:
            #     plt.imshow(new_tile)
            #     plt.show()

            new_stats += [[s, tile] for s in list(square_stats[1:])]
    else:
        if ratio > 0.7:
            # print('Filter out2', small_square)
            return

        # We have a bounding box to optimize
        # for _ in range(3):
        #     box[0, 0] -= 1
        #     box[1, 0] += 1
        #     box[2, 0] += 1
        #     box[3, 0] -= 1
        #     box[0, 1] -= 1
        #     box[1, 1] -= 1
        #     box[2, 1] += 1
        #     box[3, 1] += 1
        if w > h:
            box[0, 0] -= 1
            box[1, 0] -= 1
            box[2, 0] += 1
            box[3, 0] += 1
        else:
            box[0, 1] -= 1
            box[1, 1] += 1
            box[2, 1] += 1
            box[3, 1] -= 1

        endpoints = np.copy(box)
        # endpoints[:, 0] += x - pad
        # endpoints[:, 1] += y - pad

        # if show:
        #     plt.scatter([endpoints[0, 0]], [endpoints[0, 1]], c='r', s=.5)
        #     plt.scatter([endpoints[1, 0]], [endpoints[1, 1]], c='g', s=.5)
        #     plt.scatter([endpoints[2, 0]], [endpoints[2, 1]], c='b', s=.5)
        #     plt.scatter([endpoints[3, 0]], [endpoints[3, 1]], c='orange', s=.05)
        #     plt.gca().add_patch(Polygon(endpoints, closed=True, facecolor='g', alpha=0.6))

        polygons.append(endpoints)

    return


def process_merge(vertices, polygons, merged, close):
    a = vertices[close[0]]
    for i in close[1:]:
        if i in merged:
            continue
        b = vertices[i]
        # d = np.linalg.norm(a[0] - b[0])
        d = sqeuclidean(a[0], b[0])
        if len(a[1].intersection(b[1])) > 0 and d > 0:
            continue
        a_new = [(a[0] + b[0]) / 2.0, a[1].union(b[1])]
        # Replace b with p in polygons
        if d == 0:
            for poly_ind in a[1]:
                polygons[poly_ind] = [v for v in polygons[poly_ind] if not vertex_equal(v, a)]
                polygons[poly_ind].append(a_new)
                # merged_polygons[poly_ind] = merged_polygons[poly_ind].union(a_new[1])
        for poly_ind in b[1]:
            polygons[poly_ind] = [v for v in polygons[poly_ind] if not vertex_equal(v, b)]
            polygons[poly_ind].append(a_new)
            # if d == 0:
            # merged_polygons[poly_ind] = merged_polygons[poly_ind].union(a_new[1])
        a[0] = a_new[0]
        a[1] = a_new[1]
        merged.append(i)
        break
    return


def process_vertices(vertices, polygons, epsilon, merged, vertices_ind):
    i, a = vertices_ind
    for j, b in enumerate(vertices[i + 1:]):
        if (i + 1 + j) in merged:
            continue
        d = np.linalg.norm(a[0] - b[0])
        if d < epsilon:
            # if len(a[1].intersection(b[1])) > 0 and d > 0:
            if len(a[1].intersection(b[1])) > 0 and d > 0:
                continue
            a_new = [(a[0] + b[0]) / 2.0, a[1].union(b[1])]
            # Replace b with p in polygons
            if d == 0:
                for poly_ind in a[1]:
                    polygons[poly_ind] = [v for v in polygons[poly_ind] if not vertex_equal(v, a)]
                    polygons[poly_ind].append(a_new)
                    # merged_polygons[poly_ind] = merged_polygons[poly_ind].union(a_new[1])
            for poly_ind in b[1]:
                polygons[poly_ind] = [v for v in polygons[poly_ind] if not vertex_equal(v, b)]
                polygons[poly_ind].append(a_new)
                # if d == 0:
                # merged_polygons[poly_ind] = merged_polygons[poly_ind].union(a_new[1])
            a[0] = a_new[0]
            a[1] = a_new[1]
            merged.append(i + 1 + j)
            break


def process_poly_mp(new_tiles, final_tiles, tile_dim, img, tile):
    if len(tile) < 4:
        return
    endpoints = np.array(tile, np.int32)

    d1 = np.linalg.norm(endpoints[0] - endpoints[1])
    d2 = np.linalg.norm(endpoints[1] - endpoints[2])
    if min(d1, d2) > tile_dim:
        if d1 > d2:
            new_tiles += [
                [endpoints[0], (endpoints[0] + endpoints[1]) / 2, (endpoints[2] + endpoints[3]) / 2,
                 endpoints[3]],
                [(endpoints[0] + endpoints[1]) / 2, (endpoints[2] + endpoints[3]) / 2, endpoints[2],
                 endpoints[1]],
            ]
        else:
            new_tiles += [
                [endpoints[0], endpoints[1], (endpoints[1] + endpoints[2]) / 2,
                 (endpoints[0] + endpoints[3]) / 2],
                [(endpoints[1] + endpoints[2]) / 2, (endpoints[0] + endpoints[3]) / 2, endpoints[3],
                 endpoints[2]],
            ]
    else:
        # Determine class of polygon
        class_mask = np.zeros(img.shape, np.uint8)
        cv2.fillPoly(class_mask, [endpoints], color=1)
        class_mask *= img
        maj_class = 1
        wall_count = np.count_nonzero(class_mask == 1)
        maj_count = wall_count
        for c in dataset_classes[type][2:]:
            class_count = np.count_nonzero(class_mask == c)
            if class_count > maj_count and class_count > wall_count:
                maj_class = c
        final_tiles.append((endpoints, maj_class))
    return


def process_poly(new_tiles, tile_dim, img, result, tile):
    if len(tile) < 4:
        return
    endpoints = np.array(tile, np.int32)

    # Determine class of polygon
    class_mask = np.zeros(img.shape, np.uint8)
    cv2.fillPoly(class_mask, [endpoints], color=1)
    class_mask *= img
    class_counts = sorted([np.count_nonzero(class_mask == c) for c in dataset_classes[type][1:]])
    uncertain = class_counts[-2] >= 0.1 * class_counts[-1] or class_counts[-2] >= 3
    d1 = np.linalg.norm(endpoints[0] - endpoints[1])
    d2 = np.linalg.norm(endpoints[1] - endpoints[2])
    if min(d1, d2) > tile_dim and uncertain:
        if d1 > d2:
            new_tiles += [
                [endpoints[0], (endpoints[0] + endpoints[1]) / 2, (endpoints[2] + endpoints[3]) / 2,
                 endpoints[3]],
                [(endpoints[0] + endpoints[1]) / 2, (endpoints[2] + endpoints[3]) / 2, endpoints[2],
                 endpoints[1]],
            ]
        else:
            new_tiles += [
                [endpoints[0], endpoints[1], (endpoints[1] + endpoints[2]) / 2,
                 (endpoints[0] + endpoints[3]) / 2],
                [(endpoints[1] + endpoints[2]) / 2, (endpoints[0] + endpoints[3]) / 2, endpoints[3],
                 endpoints[2]],
            ]
    else:
        maj_class = 1
        wall_count = np.count_nonzero(class_mask == 1)
        maj_count = wall_count
        for c in dataset_classes[type][2:]:
            class_count = np.count_nonzero(class_mask == c)
            if class_count > maj_count and class_count > wall_count:
                maj_class = c
        cv2.fillPoly(result, [endpoints], color=maj_class)
    return


def process_polygons_mp(img, poly):
    if len(poly) < 3:
        return []
    points = []
    for p1 in poly:
        contains = False
        for p2 in points:
            if points_equal(p1[0], p2):
                contains = True
                break
        if not contains:
            points.append(p1[0])
    endpoints = rotational_sort(np.array(points, np.int32))

    # Split large polygons to more accurately determine class
    tiles = [endpoints]
    tile_dim = 8
    final_tiles = []
    while len(tiles) > 0:
        # print('{0} tiles left'.format(len(tiles)))
        new_tiles = []
        mp = True
        if mp:
            pool = ThreadPool(20)
            [_ for _ in pool.imap_unordered(partial(process_poly_mp, new_tiles, final_tiles, tile_dim, img),
                                            tiles)]
            pool.close()
        else:
            for tile in tiles:
                process_poly_mp(new_tiles, final_tiles, tile_dim, img, tile)
        tiles = new_tiles
    return final_tiles


def process_polygons(img, result, poly):
    if len(poly) < 3:
        return
    points = []
    for p1 in poly:
        contains = False
        for p2 in points:
            if points_equal(p1[0], p2):
                contains = True
                break
        if not contains:
            points.append(p1[0])
    endpoints = rotational_sort(np.array(points, np.int32))

    # Split large polygons to more accurately determine class
    tiles = [endpoints]
    tile_dim = 8
    while len(tiles) > 0:
        # print('{0} tiles left'.format(len(tiles)))
        new_tiles = []
        mp = False
        if mp:
            pool = ThreadPool(2)
            [_ for _ in pool.imap_unordered(partial(process_poly, new_tiles, tile_dim, img, result),
                                            tiles)]
            pool.close()
        else:
            for tile in tiles:
                process_poly(new_tiles, tile_dim, img, result, tile)
        tiles = new_tiles


def refine_clusters(img, high_res, debug):
    org_shape = img.shape
    if high_res:
        scale = 3
        new_d = int(scale * img.shape[0])
        # img = cv2.resize(img, [new_d, new_d], interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, [new_d, new_d], interpolation=cv2.INTER_NEAREST)

    show = False
    generate = True
    if generate:
        polygons = []

        # TODO eval this
        # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3)))

        img = img.astype(np.uint8)
        _, _, stats, centroids = cv2.connectedComponentsWithStats(img)

        stats = [[s, None] for s in list(stats[1:])]
        initial_len = len(stats)

        while len(stats) > 0:
            if debug:
                print('{0} left'.format(len(stats)))
            new_stats = []
            mp = True
            if mp:
                # pool = ThreadPool(15)
                pool = ThreadPool(30)
                [_ for _ in pool.imap_unordered(partial(process_tile, initial_len, img, False, polygons, new_stats),
                                                enumerate(stats))]
                pool.close()
            else:
                plt.figure(dpi=300)
                plt.imshow(img > 0)
                plt.axis('off')
                plt.savefig('results/pp_joint_mask', bbox_inches='tight', pad_inches=0)
                plt.show()

                for d in enumerate(stats):
                    # process_tile(initial_len, img, False, polygons, new_stats, d)
                    process_tile(initial_len, img, True, polygons, new_stats, d)
            stats = new_stats
            # for s in new_stats:
            #     stats += s[0]
            #     polygons += s[1]
            initial_len = -1

        if debug:
            np.save('results/pp', polygons)

    # vertex structure: [p1, p2, p3, p4]
    if debug:
        polygons = np.load('results/pp.npy')
        print(len(polygons))

    if show:
        bg = np.zeros(img.shape)
        plt.figure(dpi=300)
        plt.imshow(bg)
        for endpoints in polygons:
            plt.scatter([endpoints[0, 0]], [endpoints[0, 1]], c='r', s=.1)
            # plt.scatter([endpoints[1, 0]], [endpoints[1, 1]], c='g', s=.1)
            # plt.scatter([endpoints[2, 0]], [endpoints[2, 1]], c='b', s=.1)
            plt.scatter([endpoints[3, 0]], [endpoints[3, 1]], c='orange', s=.1)
            plt.gca().add_patch(Polygon(endpoints, closed=True, facecolor='g', alpha=0.6))
        plt.show()

    epsilon = 4
    vertices = []
    for i, p in enumerate(polygons):
        vertices += [[v, {i}] for v in p]
    polygons = []
    for i in range(0, len(vertices), 4):
        polygons.append(vertices[i: i + 4])

    # merged_polygons = [set() for _ in polygons]

    # Optimize distance
    if debug:
        print('{0} vertices'.format(len(vertices)))
    tic = time.time()
    while True:
        merged = []
        coords = np.array([v[0] for v in vertices])
        tree = KDTree(coords)
        merges = dict()
        rows = tree.query_pairs(r=epsilon)
        for l, r in rows:
            if l not in merges:
                merges[l] = [l]
            merges[l].append(r)

        mp = True
        if mp:
            pool = ThreadPool(30)
            [_ for _ in pool.imap_unordered(partial(process_merge, vertices, polygons, merged),
                                            merges.values())]
            pool.close()
        else:
            for close in merges.values():
                process_merge(vertices, polygons, merged, close)

        # pool = ThreadPool(30)
        # [_ for _ in pool.imap_unordered(partial(process_vertices, vertices, polygons, epsilon, merged),
        #                                 enumerate(vertices))]
        # pool.close()
        #
        # print('{0} merged'.format(len(merged)))
        if len(merged) == 0:
            break
        else:
            vertices = [value for (i, value) in enumerate(vertices) if i not in merged]

    toc = time.time()
    if debug:
        print('Done merging close vertices: ', (toc - tic) / 60)
        print('{0} vertices'.format(len(vertices)))

    # print(merged_polygons)
    # for _ in range(100):
    #     for i in range(len(merged_polygons)):
    #         merged_polygons[i] = merged_set(merged_polygons[i], merged_polygons)
    if show:
        plot_polygons(img.shape, vertices, polygons)
    # exit(0)

    # Optimize polygons
    # while True:
    #     merge = False
    #     for v in vertices:
    #         for i, polygon in enumerate(polygons):
    #             if i in v[1]:
    #                 continue
    #             endpoints = rotational_sort(np.stack(np.array(polygon)[:, 0], axis=0))
    #             path = Polygon(endpoints, closed=True).get_path()
    #             r = 0.01
    #             if path.contains_point(v[0], radius=r) or path.contains_point(v[0], radius=-r):
    #                 print('merge', v, polygon)
    #                 v_new = [v[0], v[1].union({i})]
    #                 for poly_ind in v[1]:
    #                     polygons[poly_ind] = [v2 for v2 in polygons[poly_ind] if
    #                                           not vertex_equal(v2, v)]
    #                     polygons[poly_ind].append(v_new)
    #                 v[1] = v_new[1]
    #                 polygon.append(v)
    #                 merge = True
    #     if not merge:
    #         break
    # print('Done optimizing polygons')
    # plot_polygons(bd_ind.shape, vertices, polygons)

    # Optimize angle
    tic = time.time()
    while True:
        merged = False
        for polygon in polygons:
            # if merged:
            #     break
            merged = False
            for p1 in polygon:
                if merged:
                    break
                # Loop through polygons
                for p1_poly_ind in p1[1]:
                    if merged:
                        break
                    for p2 in polygons[p1_poly_ind]:
                        if merged:
                            break
                        if points_equal(p1[0], p2[0]):
                            continue
                        # Loop through polygons
                        for p2_poly_ind in p2[1]:
                            if merged:
                                break
                            for p3 in polygons[p2_poly_ind]:
                                if points_equal(p1[0], p3[0]) or points_equal(p2[0], p3[0]):
                                    continue
                                # Derive a,b,c in order s.t. a->b->c
                                a = [[1e6, 1e6], []]
                                c = [[-1, -1], []]
                                for d in [p1, p2, p3]:
                                    if sum(d[0]) < sum(a[0]):
                                        a = d
                                    if sum(d[0]) > sum(c[0]):
                                        c = d
                                for d in [p1, p2, p3]:
                                    if not points_equal(d[0], a[0]) and not points_equal(d[0], c[0]):
                                        b = d
                                # print(a, b, c, angle_between_three_points(a[0], b[0], c[0]))
                                angle = angle_between_three_points(a[0], b[0], c[0])
                                if angle > 0.97029:
                                    if a[1] != c[1] or a[1] != b[1]:
                                        # if merged_set(a[1], merged_polygons) != merged_set(c[1], merged_polygons):
                                        continue
                                    a_new = [a[0], b[1]]
                                    c_new = [c[0], b[1]]
                                    # a_new = [a[0], a[1].union(b[1])]
                                    # c_new = [c[0], c[1].union(b[1])]
                                    if a[1] != c[1]:
                                        # Add c to a polygons
                                        for poly_ind in a[1]:
                                            polygons[poly_ind].append(c_new)
                                        # Add a to c polygons
                                        for poly_ind in c[1]:
                                            polygons[poly_ind].append(a_new)
                                        a[1] = b[1]
                                        c[1] = b[1]

                                    # Remove b from polygons
                                    for b_poly_ind in b[1]:
                                        polygons[b_poly_ind] = [v for v in polygons[b_poly_ind] if
                                                                not vertex_equal(v, b)]
                                    merged = True
                                    break
        if not merged:
            break

    toc = time.time()
    if debug:
        print('Done merging angle vertices: ', (toc - tic) / 60)

    result = np.zeros(img.shape, np.uint8)

    if debug:
        print('Refining {0} polygons'.format(len(polygons)))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5)))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, np.ones((3, 3)))
    #
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3)))
    tic = time.time()
    mp = True
    if mp:
        # pool = Pool(processes=15)
        # res = pool.map(partial(process_polygons_mp, img),
        #                polygons)
        # pool.close()
        # pool.join()
        pool = ThreadPool(50)
        [_ for _ in pool.imap_unordered(partial(process_polygons, img, result),
                                        polygons)]
        pool.close()
    else:
        i = 1
        for poly in polygons:
            print(len(polygons) - i)
            i += 1
            process_polygons(img, result, poly)
    toc = time.time()
    if debug:
        print('Done refining polygons: ', (toc - tic) / 60)

    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((5, 5)))

    # result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, np.ones((3, 3)), iterations=3)
    # result = cv2.morphologyEx(result, cv2.MORPH_ERODE, np.ones((3, 3)))
    # result = cv2.morphologyEx(result, cv2.MORPH_DILATE, np.ones((3, 3)))

    if high_res:
        result = cv2.resize(result, org_shape, interpolation=cv2.INTER_NEAREST)
        # result = cv2.resize(result, org_shape, interpolation=cv2.INTER_AREA)

    return result


def apply_heuristics(img, dataset):
    # Set labels
    #         'walls': 1,
    #         'glass_walls': 2,
    #         'railings': 3,
    #         'doors': 4,
    #         'sliding_doors': 5,
    #         'windows': 6,
    #         'stairs': 7,
    wall_n = dataset_classes_map[dataset]['walls']
    glass_wall_n = dataset_classes_map[dataset]['glass_walls']
    railing_n = dataset_classes_map[dataset]['railings']
    door_n = dataset_classes_map[dataset]['doors']
    sliding_door_n = dataset_classes_map[dataset]['sliding_doors']
    window_n = dataset_classes_map[dataset]['windows']
    stairs_n = dataset_classes_map[dataset]['stairs']

    # Apply heuristics
    ymax, xmax = img.shape

    # Remove railings next to windows
    # updates = 0
    # railings = np.argwhere(img == railing_n)
    # for y,x in railings:
    #     neighbours = img[max(y-1,0):min(y+2,ymax), max(x-1,0):min(x+2,xmax)].flatten()
    #     railings_count = len([n for n in neighbours if n == railing_n])
    #     windows_count = len([n for n in neighbours if n == window_n])
    #     if windows_count > 0.2 * railings_count:
    #         updates += 1
    #         img[y,x] = 6
    # print('{0} fixes'.format(updates))

    # Replace glass walls or windows depending on majority in neighbours
    updates = 0
    glass_walls = np.argwhere(img == glass_wall_n)
    for y, x in glass_walls:
        neighbours = img[max(y - 1, 0):min(y + 2, ymax), max(x - 1, 0):min(x + 2, xmax)].flatten()
        glass_walls_count = len([n for n in neighbours if n == glass_wall_n])
        windows_count = len([n for n in neighbours if n == window_n])
        if windows_count > glass_walls_count:
            updates += 1
            img[y, x] = window_n
    windows = np.argwhere(img == window_n)
    for y, x in windows:
        neighbours = img[max(y - 1, 0):min(y + 2, ymax), max(x - 1, 0):min(x + 2, xmax)].flatten()
        glass_walls_count = len([n for n in neighbours if n == glass_wall_n])
        windows_count = len([n for n in neighbours if n == window_n])
        if windows_count <= glass_walls_count and img[y, x] != glass_wall_n:
            updates += 1
            img[y, x] = glass_wall_n
    # print('{0} glasswalls/windows fixes'.format(updates))

    # Replace doors or sliding doors depending on majority in neighbours
    updates = 0
    doors = np.argwhere(img == door_n)
    for y, x in doors:
        neighbours = img[max(y - 1, 0):min(y + 2, ymax), max(x - 1, 0):min(x + 2, xmax)].flatten()
        doors_count = len([n for n in neighbours if n == door_n])
        sliding_doors_count = len([n for n in neighbours if n == sliding_door_n])
        if doors_count < sliding_doors_count:
            updates += 1
            img[y, x] = sliding_door_n
    sliding_doors = np.argwhere(img == sliding_door_n)
    for y, x in sliding_doors:
        neighbours = img[max(y - 1, 0):min(y + 2, ymax), max(x - 1, 0):min(x + 2, xmax)].flatten()
        doors_count = len([n for n in neighbours if n == door_n])
        sliding_doors_count = len([n for n in neighbours if n == sliding_door_n])
        if doors_count >= sliding_doors_count and img[y, x] != door_n:
            updates += 1
            img[y, x] = door_n
    # print('{0} doors fixes'.format(updates))

    # Replace non-background pixels neighbouring stairs with walls/railings
    for _ in range(2):
        updates = 0
        stairs = np.argwhere(img == stairs_n)
        for y, x in stairs:
            ksize = 3
            neighbours = img[max(y - ksize, 0):min(y + ksize + 1, ymax),
                         max(x - ksize, 0):min(x + ksize + 1, xmax)].flatten()
            nonbg_count = len([n for n in neighbours if (n != stairs_n and n > 0)])
            if nonbg_count == 0:
                continue
            # walls_count = len([n for n in neighbours if n == wall_n])
            # railings_count = len([n for n in neighbours if n == railing_n])
            # replace_n = wall_n if walls_count > railings_count else railing_n
            for y1 in range(max(y - ksize, 0), min(y + ksize + 1, ymax)):
                for x1 in range(max(x - ksize, 0), min(x + ksize + 1, xmax)):
                    ksize = 4
                    neighbours = img[max(y1 - ksize, 0):min(y1 + ksize + 1, ymax),
                                 max(x1 - ksize, 0):min(x1 + ksize + 1, xmax)].flatten()
                    walls_count = len([n for n in neighbours if n == wall_n])
                    railings_count = len([n for n in neighbours if n == railing_n])
                    replace_n = wall_n if walls_count > railings_count else railing_n
                    # replace_n = wall_n
                    # print(replace_n)
                    if img[y1, x1] != 0 and img[y1, x1] != stairs_n and img[y1, x1] != replace_n:
                        updates += 1
                        img[y1, x1] = replace_n
        print('{0} stairs fixes'.format(updates))

    return img


def post_process(bd_ind, high_res=True, eval=False, debug=False, zeng=False):
    dataset = 'multi'
    # ignore the background mislabeling
    # result = fill_break_line(bd_ind)

    # plt.imshow(bd_ind)
    # plt.show()
    tic = time.time()
    result = bd_ind
    if zeng:
        result1 = fill_break_line(result)
    else:
        result1 = refine_clusters(result, high_res, debug)
    result2 = apply_heuristics(result1, dataset)
    toc = time.time()
    print('Done: ', (toc - tic) / 60)
    if eval:
        return result1, result2
    else:
        return result2
