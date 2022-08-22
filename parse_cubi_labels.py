import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def draw(points):
    coord = []
    for point in points.split(' ')[:-1]:
        if point == '':
            continue
        x, y = point.split(',')
        x = float(x)
        y = float(y)
        coord.append([x, y])

    coord.append(coord[0])
    return coord


def circle(circle):
    return [(float(circle['cx']), float(circle['cy'])), float(circle['r'])]


def parseFloor(floor, elements):
    for child in floor[0]:
        if 'id' in child.attrib:
            if child.attrib['id'] == 'Railing':
                assert len(child) == 1  # Railing consists of a single polygon
                assert 'points' in child[0].attrib
                elements[child.attrib['id']].append(draw(child[0].attrib['points']))
            elif child.attrib['id'] == 'Wall':
                elements[child.attrib['id']].append(draw(child[0].attrib['points']))
                for c in child[1:]:
                    assert c.attrib['id'] == 'Door' or c.attrib['id'] == 'Window'
                    elements[c.attrib['id']].append(draw(c[0].attrib['points']))
            elif child.attrib['id'] == 'Column':
                assert len(child) == 1  # Railing consists of a single polygon or circle
                if 'Circle' in child.attrib['class']:
                    # Handle circles
                    assert 'cx' in child[0].attrib
                    elements[child.attrib['id']].append(circle(child[0].attrib))
                else:
                    assert 'points' in child[0].attrib
                    elements['Wall'].append(draw(child[0].attrib['points']))
            elif child.attrib['id'] == 'Stairs':
                for c in child:
                    elements[child.attrib['id']].append(draw(c[0].attrib['points']))
            else:
                # print(child.attrib['id'])
                pass


def parse_cubi_label(dir):
    for path in [dir]:
        xml = ET.parse(path + '/model.svg')
        root = xml.getroot()

        classes = ['Stairs', 'Railing', 'Wall', 'Window', 'Door', 'Column']
        elements = {}
        for c in classes:
            elements[c] = []

        for floor in root.find('{http://www.w3.org/2000/svg}g'):
            if floor.attrib['class'] == 'Floor':
                parseFloor(floor, elements)

        h, w, _ = cv2.imread(path + '/F1_scaled.png').shape
        bg = np.zeros((h, w))

        show = True
        i = 0
        if show:
            for c in classes:
                plt.figure(dpi=600)
                plt.imshow(bg, cmap='gray')
                for p in elements[c]:
                    if i == len(classes) - 1:
                        # Handle circles
                        plt.gca().add_patch(Circle(p[0], p[1], color='white', linewidth=0))
                    else:
                        xs, ys = zip(*p)
                        plt.fill(xs, ys, 'white')
                i += 1

                plt.axis('off')
                plt.savefig(path + '/' + c.lower() + '.png', bbox_inches='tight', pad_inches=0)
                # plt.show()
                plt.close()


if __name__ == '__main__':
    main()
