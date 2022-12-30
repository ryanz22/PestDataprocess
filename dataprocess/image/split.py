from collections import Counter
from PIL import Image
import os


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


def split_image(image_path, rows, cols, should_square, should_cleanup, should_quiet=False, output_dir=None):
    im = Image.open(image_path)
    im_width, im_height = im.size
    row_width = int(im_width / cols)
    row_height = int(im_height / rows)
    name, ext = os.path.splitext(image_path)
    name = os.path.basename(name)
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_dir = "./"
    if should_square:
        min_dimension = min(im_width, im_height)
        max_dimension = max(im_width, im_height)
        if not should_quiet:
            print("Resizing image to a square...")
            print("Determining background color...")
        bg_color = determine_bg_color(im)
        if not should_quiet:
            print("Background color is... " + str(bg_color))
        im_r = Image.new("RGBA" if ext == "png" else "RGB",
                         (max_dimension, max_dimension), bg_color)
        offset = int((max_dimension - min_dimension) / 2)
        if im_width > im_height:
            im_r.paste(im, (0, offset))
        else:
            im_r.paste(im, (offset, 0))
        if not should_quiet:
            print("Exporting resized image...")
        outp_path = name + "_squared" + ext
        outp_path = os.path.join(output_dir, outp_path)
        im_r.save(outp_path)
        im = im_r
        row_width = int(max_dimension / cols)
        row_height = int(max_dimension / rows)
    n = 0
    for i in range(0, rows):
        for j in range(0, cols):
            box = (j * row_width, i * row_height, j * row_width +
                   row_width, i * row_height + row_height)
            outp = im.crop(box)
            outp_path = name + "_" + str(n) + ext
            outp_path = os.path.join(output_dir, outp_path)
            if not should_quiet:
                print("Exporting image tile: " + outp_path)
            outp.save(outp_path)
            n += 1
    if should_cleanup:
        if not should_quiet:
            print("Cleaning up: " + image_path)
        os.remove(image_path)


def reverse_split(paths_to_merge, rows, cols, image_path, should_cleanup, should_quiet=False):
    if len(paths_to_merge) == 0:
        print("No images to merge!")
        return
    for index, path in enumerate(paths_to_merge):
        path_number = int(path.split("_")[-1].split(".")[0])
        if path_number != index:
            print("Warning: Image " + path +
                  " has a number that does not match its index!")
            print("Please rename it first to match the rest of the images.")
            return
    images_to_merge = [Image.open(p) for p in paths_to_merge]
    image1 = images_to_merge[0]
    new_width = image1.size[0] * cols
    new_height = image1.size[1] * rows
    print(paths_to_merge)
    new_image = Image.new(image1.mode, (new_width, new_height))
    if not should_quiet:
        print("Merging image tiles with the following layout:", end=" ")
    for i in range(0, rows):
        print("\n")
        for j in range(0, cols):
            print(paths_to_merge[i * cols + j], end=" ")
    print("\n")
    for i in range(0, rows):
        for j in range(0, cols):
            image = images_to_merge[i * cols + j]
            new_image.paste(image, (j * image.size[0], i * image.size[1]))
    if not should_quiet:
        print("Saving merged image: " + image_path)
    new_image.save(image_path)
    if should_cleanup:
        for p in paths_to_merge:
            if not should_quiet:
                print("Cleaning up: " + p)
            os.remove(p)


def determine_bg_color(im):
    im_width, im_height = im.size
    rgb_im = im.convert('RGBA')
    all_colors = []
    areas = [[(0, 0), (im_width, im_height / 10)],
             [(0, 0), (im_width / 10, im_height)],
             [(im_width * 9 / 10, 0), (im_width, im_height)],
             [(0, im_height * 9 / 10), (im_width, im_height)]]
    for area in areas:
        start = area[0]
        end = area[1]
        for x in range(int(start[0]), int(end[0])):
            for y in range(int(start[1]), int(end[1])):
                pix = rgb_im.getpixel((x, y))
                all_colors.append(pix)
    return Counter(all_colors).most_common(1)[0][0]
