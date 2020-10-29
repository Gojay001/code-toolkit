# x, y is in (minx, miny, maxx, maxy)

def bbox_iou(x, y):
	left = max(x[0], y[0])
	top = max(x[1], y[1])
	right = min(x[2], y[2])
	bottom = min(x[3], y[3])

	if left >= right or top >= bottom:
		return 0

	inter = (right - left) * (bottom - top)
	calc_area = lambda pos: (pos[2] - pos[0]) * (pos[3] - pos[1])
	area_x = calc_area(x)
	area_y = calc_area(y)
	iou = inter / float(area_x + area_y - inter)
	return iou

if __name__ == '__main__':
	pos1, pos2 = [1, 1, 5, 5], [4, 4, 7, 7]
	iou = bbox_iou(pos1, pos2)
	print(iou)