def overlap_split(
    img,
    split_width: int = 150,
    split_height: int = 150,
    overlap: float = 0.5,
):
    """split image with overlap"""
    img_h, img_w, _ = img.shape

    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)

    img_list = []

    for i in Y_points:
        for j in X_points:
            split = img[i : i + split_height, j : j + split_width]
            img_list.append(split)

    return img_list


def start_points(size: int, split_size: int, overlap: float = 0.0):
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            if split_size == size:
                break
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points
